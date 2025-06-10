import itertools
import logging
import os
import pickle
import shutil
from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

import engine.config as config_utils
from graphs.frag_graph import FragGraph
from models import constants
from seq import frags


class GraphDataset:
    DATASET_FEATURES = ['min_weight', 'max_weight', 'n_edges', 'density', 'n_articulation_points',
                        'diameter', 'n_variants', 'compression_factor']
    GLOBAL_FIELDS = ['Global', 'shuffle']

    def __init__(self, config):
        self.config = config
        self.features = list(
            feature for feature_name in config.features for feature in constants.FEATURES_DICT[feature_name])

        self.panel = config.panel
        shutil.copyfile(config.panel, config.experiment_dir + '/panel.txt')

        self.filter_metrics = set()
        if config.filter_config:
            self.filter_config = config_utils.load_config(config.filter_config,
                                                          config_type=config_utils.CONFIG_TYPE.DATA_FILTER)
            # Copy the filter used in the local folder
            with open(config.experiment_dir + '/filters.yaml', 'w') as outfile:
                yaml.safe_dump(self.filter_config.__dict__, outfile, default_flow_style=False)

            # Determine the features to compute to perform the data filtering and ordering
            for group, filters in vars(self.filter_config).items():
                if group == 'shuffle': continue
                for metric in filters:
                    if metric in ['size_train', 'size_validate']: continue
                    self.filter_metrics.add(metric)

    @staticmethod
    def round_robin_validation(n_procs, dataset):
        size_ordered = dataset.sort_values(by=['n_nodes'], ascending=True)
        component_chunks = []
        for i in range(min(n_procs, len(size_ordered))):
            component_chunks.append(size_ordered.iloc[i:: n_procs, :])
        return component_chunks

    @staticmethod
    def filter_dataset(dataset, global_dataset, filters):
        def apply_filter(dataset_to_filter, metric, lower_bound, upper_bound):
            if lower_bound:
                if upper_bound and upper_bound < lower_bound:
                    logging.info(
                        f'WARNING: the lower bound {lower_bound} is greater than the upper bound {upper_bound} for {metric}.')
                dataset_to_filter = dataset_to_filter[dataset_to_filter[metric] >= lower_bound]
            if upper_bound:
                dataset_to_filter = dataset_to_filter[dataset_to_filter[metric] <= upper_bound]
            return dataset_to_filter

        filtered_graphs = dataset.copy()
        # Quantiles have to be computed from the original dataset after global filters to be accurate
        for filter_name, filter_dict in filters.items():
            if filter_name in ['size_train', 'size_validate']: continue
            if ("min" in filter_dict) or ("max" in filter_dict):
                if "quantiles" in filter_dict:
                    logging.info(
                        f'WARNING: both quantiles and min/max provided {filters}, quantiles will be ignored.')
                min_val = None
                max_val = None
                if "min" in filter_dict:
                    min_val = filter_dict["min"]
                if "max" in filter_dict:
                    max_val = filter_dict["max"]
                filtered_graphs = apply_filter(filtered_graphs, filter_name, min_val, max_val)
            elif "quantiles" in filter_dict:
                quantiles = filter_dict["quantiles"]
                buckets = global_dataset[filter_name].quantile(quantiles)
                filtered_graphs = apply_filter(filtered_graphs, filter_name, buckets[filter_dict["quantiles"][0]],
                                                 buckets[filter_dict["quantiles"][1]])
            else:
                raise(NotImplementedError, 'Please use a filter using min/max or quantiles.')
        return filtered_graphs

    def dataset_nested_design(self, dataset):
        # parses the nested filter_config.yaml, which allows arbitrary specifications
        # of training/validation set design for any combination of graph metrics
        dataset = self.filter_dataset(dataset, dataset, self.filter_config.Global)
        training_dataset = []
        validation_dataset = []
        global_dataset = dataset.copy()
        for group, group_dict in vars(self.filter_config).items():
            if group in self.GLOBAL_FIELDS: continue
            filtered_dataset = self.filter_dataset(dataset, global_dataset, group_dict)
            logging.info("Selected %d graphs for group %s." % (filtered_dataset.shape[0], group))
            size_train = self.config.size_train
            if "size_train" in group_dict:
                size_train = group_dict["size_train"]
            size_validate = self.config.size_validate
            if "size_validate" in group_dict:
                size_validate = group_dict["size_validate"]
            # Select the validation graphs and remove them from the available graphs
            validation_graphs, filtered_dataset = self.sample_datasets(filtered_dataset, size_validate, 'validation', group+' ')
            dataset = dataset[~dataset.component_path.isin(validation_graphs.component_path)]
            validation_graphs["group"] = group
            validation_dataset.append(validation_graphs)

            # Select the training graphs from the remain graphs, if size_train is None, all remaining selected graphs are kept for training
            training_graphs, _ = self.sample_datasets(filtered_dataset, size_train, 'train', group+' ')

            dataset = dataset[~dataset.component_path.isin(training_graphs.component_path)]
            training_graphs["group"] = group
            training_dataset.append(training_graphs)

        training_dataset = pd.concat(training_dataset)
        if self.filter_config.shuffle:
            training_dataset = training_dataset.sample(frac=1, random_state=self.config.seed)

        validation_dataset = pd.concat(validation_dataset)

        return training_dataset, validation_dataset

    @staticmethod
    def load_dataset(panel):
        # Merge the datasets whose paths are listed in the panel file.
        datasets = []
        if isinstance(panel, str):
            panel = [panel]
        if not isinstance(panel, list):
            raise ValueError(f'The panel must be a path or a list of paths, provided {panel}')
        for dataset in panel:
            path_dataset = dataset.strip()
            datasets.append(pd.read_pickle(path_dataset))
        graph_dataset = pd.concat(datasets)
        logging.info("graph dataset... %s" % graph_dataset.describe())
        return graph_dataset

    def compute_dataset_metrics(self, frag_graph):
        metric_dict = {}
        for metric in self.filter_metrics:
            if metric == "n_edges":
                metric_dict["n_edges"] = frag_graph.g.number_of_edges()
            elif metric == "n_nodes":
                metric_dict["n_nodes"] = frag_graph.g.number_of_nodes()
            elif metric == "density":
                metric_dict["density"] = nx.density(frag_graph.g)
            elif metric == "min_weight":
                edge_labels = nx.get_edge_attributes(frag_graph.g, "weight")
                metric_dict["min_weight"] = min(edge_labels.values())
            elif metric == "max_weight":
                edge_labels = nx.get_edge_attributes(frag_graph.g, "weight")
                metric_dict["max_weight"] = max(edge_labels.values())
            elif metric == "n_articulation_points":
                metric_dict["n_articulation_points"] = len(list(nx.articulation_points(frag_graph.g)))
            elif metric == "diameter":
                metric_dict["diameter"] = nx.diameter(frag_graph.g)
            elif metric == "n_variants":
                metric_dict['n_variants'] = frag_graph.get_variants_set()
            elif metric == "compression_factor":
                total_n_frags = sum([frag.n_copies for frag in frag_graph.fragments])
                metric_dict['compression_factor'] = 1 - frag_graph.g.number_of_nodes() / total_n_frags
            else:
                raise ValueError(f'Metric {metric} not defined, please use one of {self.DATASET_FEATURES}.')
        return metric_dict

    def is_invalid_subgraph(self, subgraph):
        return subgraph.n_nodes == 1 and self.config.skip_singleton_graphs

    def generate_graphs(self, chunk):
        # Get the path of the parent folder of the panel file
        config = deepcopy(self.config)
        logging.root.setLevel(logging.getLevelName(config.logging_level))
        frag_folder = self.config.out_dir
        combined_graph_indexes = []
        for i in range(len(chunk)):
            # precompute graphs and get their feature distributions
            config.bam = chunk[i]['bam_path']
            config.vcf = chunk[i]['vcf_path']
            chromosome = chunk[i]['chromosome']
            if config.drop_chr and chromosome in config.drop_chr: continue
            logging.info("Building graphs for %s %s" % (config.bam, chromosome))
            fragments = frags.generate_fragments(config, chromosome)
            fragments = frags.parse_frag_repr(fragments)
            graph = FragGraph.build(fragments, compress=self.config.compress)
            connected_components = graph.connected_components_subgraphs(config, self.features,
                                                                        skip_trivial_graphs=self.config.skip_trivial_graphs)
            for component_index, component in enumerate(connected_components):
                if self.is_invalid_subgraph(component): continue
                component_path = frag_folder + "/" + config.bam.split('/')[-1] + "_" + chromosome + "_" + str(component_index)
                with open(component_path, 'wb') as f:
                    pickle.dump(component, f)
                metrics = self.compute_dataset_metrics(component)
                metrics['component_path'] = component_path
                metrics['chromosome'] = chromosome
                metrics['index'] = component_index
                if 'n_nodes' not in metrics:
                    metrics['n_nodes'] = component.n_nodes
                combined_graph_indexes.append(metrics)
            logging.info("Finished building graphs for %s %s" % (config.bam, chromosome))
        return combined_graph_indexes

    def sample_datasets(self, dataset, num_samples, dataset_type, group=''):
        if not num_samples:
            num_samples = dataset.shape[0]
        selected_graphs = dataset.sample(n=min(num_samples, dataset.shape[0]), random_state=self.config.seed)
        remaining_graphs = dataset[~dataset.component_path.isin(selected_graphs.component_path)]

        if num_samples > dataset.shape[0]:
            logging.info(f'WARNING: {group}not enough graphs to build the {dataset_type} set, kept {dataset.shape[0]}.')
        logging.info("%s%s Dataset contains %d graphs." % (group, dataset_type, selected_graphs.shape[0]))
        return selected_graphs, remaining_graphs

    def generate_training_validation_datasets(self):
        """
        generate dataframe containing path of each graph (saved as a FragGraph object),
         as well as pre-computed statistics about the graph such as connectivity, size, density etc.
        """
        assert os.path.exists(self.panel)
        panel = open(self.panel, 'r').readlines()
        # Check if the panel combines different panels
        chr_chunks = np.array([{'bam_path': panel[i].strip().split()[0], 'vcf_path': panel[i].strip().split()[1],
                            'chromosome': self.config.chr_names[j]}
                           for j in range(len(self.config.chr_names)) for i in range(len(panel))])
        chr_chunks = np.array_split(chr_chunks, self.config.n_procs)
        logging.info("Running on %d processes" % self.config.n_procs)
        logging.info("Chromosomes/process partition: " + str([np.array2string(chk) for chk in chr_chunks]))
        output = Parallel(n_jobs=self.config.n_procs)(
            delayed(self.generate_graphs)(chr_chunks[i]) for i in range(self.config.n_procs))
        combined_graph_indexes = pd.DataFrame(list(itertools.chain.from_iterable(output)))
        logging.info("Generated %d graphs." % combined_graph_indexes.shape[0])

        logging.info("Starting Graph Selection")
        if self.config.filter_config is not None:
            training_dataset, validation_dataset = self.dataset_nested_design(combined_graph_indexes)
            logging.info("Training Dataset contains %d graphs." % (training_dataset.shape[0]))
            logging.info("Validation Dataset contains %d graphs." % (validation_dataset.shape[0]))
        else:
            validation_dataset, remaining_graphs = self.sample_datasets(combined_graph_indexes, self.config.size_validate, 'validation')
            validation_dataset["group"] = 'Global'
            training_dataset, _ = self.sample_datasets(remaining_graphs, self.config.size_train, 'train')
            training_dataset["group"] = 'Global'
        training_dataset.to_pickle(self.config.experiment_dir + '/train.index_per_graph')
        validation_dataset.to_pickle(self.config.experiment_dir + '/validate.index_per_graph')

        logging.info("Finished Building Datasets")
