import itertools
import logging
import os
import pickle
import shutil
from copy import deepcopy
import networkx as nx
import numpy as np
import pandas as pd
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

        self.selection_config = None
        self.selection_metrics = set()
        if config.selection_config:
            self.selection_config = config.selection_config
            # Determine the features to compute to perform the data selection and ordering
            for subset_id, selection_params in vars(self.config.selection_config).items():
                if subset_id == 'shuffle': continue
                for metric in selection_params:
                    if metric in ['size_train', 'size_validate']: continue
                    self.selection_metrics.add(metric)
        self.generate_training_validation_datasets()

    @staticmethod
    def round_robin_validation(n_procs, validation_dataset):
        # Split the validation dataframe into chunks for parallelization
        # The graphs are ordered by size to distribute large graphs into the chunks
        size_ordered_dataset = validation_dataset.sort_values(by=['n_nodes'], ascending=True)
        validation_chunks = []
        for i in range(min(n_procs, len(size_ordered_dataset))):
            validation_chunks.append(size_ordered_dataset.iloc[i:: n_procs, :])
        return validation_chunks

    @staticmethod
    def select_subset(dataset, global_dataset, rules):
        def apply_selection(selected_dataset, metric, lower_bound, upper_bound):
            if lower_bound:
                if upper_bound and upper_bound < lower_bound:
                    logging.info(
                        f'WARNING: the lower bound {lower_bound} is greater than the upper bound {upper_bound} for {metric}.')
                selected_dataset = selected_dataset[selected_dataset[metric] >= lower_bound]
            if upper_bound:
                selected_dataset = selected_dataset[selected_dataset[metric] <= upper_bound]
            return selected_dataset

        selected_graphs = dataset.copy()
        # Quantiles have to be computed from the original dataset after global selection to be accurate
        for metric_name, selection_params in rules.items():
            if metric_name in ['size_train', 'size_validate']: continue
            if ("min" in selection_params) or ("max" in selection_params):
                if "quantiles" in selection_params:
                    logging.info(
                        f'WARNING: both quantiles and min/max provided {rules}, quantiles will be ignored.')
                min_val = None
                max_val = None
                if "min" in selection_params:
                    min_val = selection_params["min"]
                if "max" in selection_params:
                    max_val = selection_params["max"]
                selected_graphs = apply_selection(selected_graphs, metric_name, min_val, max_val)
            elif "quantiles" in selection_params:
                quantiles = selection_params["quantiles"]
                quantile_bounds = global_dataset[metric_name].quantile(quantiles)
                selected_graphs = apply_selection(selected_graphs, metric_name, quantile_bounds[selection_params["quantiles"][0]],
                                                 quantile_bounds[selection_params["quantiles"][1]])
            else:
                raise(NotImplementedError, f'Please define selection rules using min/max or quantiles, {selection_params} provided.')
        return selected_graphs

    def select_graphs(self, dataset):
        # parses the selection_config.yaml, for a refined selection
        # of training/validation graphs using pre-computed metrics
        dataset = self.select_subset(dataset, dataset, self.selection_config.Global)
        training_dataset = []
        validation_dataset = []
        global_dataset = dataset.copy()
        for subset_id, select_params in vars(self.selection_config).items():
            if subset_id in self.GLOBAL_FIELDS: continue
            remaining_graphs = self.select_subset(dataset, global_dataset, select_params)
            logging.info("Selected %d graphs for group %s." % (remaining_graphs.shape[0], subset_id))
            size_train = None
            if "size_train" in select_params:
                size_train = select_params["size_train"]
            size_validate = self.config.validation_ratio
            if "size_validate" in select_params:
                size_validate = select_params["size_validate"]
            # Select the validation graphs and remove them from the available graphs
            validation_graphs, remaining_graphs = self.sample_graphs(remaining_graphs, size_validate, 'validation', subset_id)
            dataset = dataset[~dataset.component_path.isin(validation_graphs.component_path)]
            validation_dataset.append(validation_graphs)

            # Select the training graphs from the remain graphs, if size_train is None, all remaining selected graphs are kept for training
            training_graphs, _ = self.sample_graphs(remaining_graphs, size_train, 'train', subset_id)

            dataset = dataset[~dataset.component_path.isin(training_graphs.component_path)]
            training_dataset.append(training_graphs)

        training_dataset = pd.concat(training_dataset)
        if self.selection_config.shuffle:
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
        graph_dataset = pd.concat(datasets, ignore_index=True)
        return graph_dataset

    def compute_dataset_metrics(self, frag_graph):
        metric_dict = {}
        for metric in self.selection_metrics:
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

    def generate_graphs(self, path_list):
        # Get the path of the parent folder of the panel file
        config = deepcopy(self.config)
        logging.root.setLevel(logging.getLevelName(config.logging_level))
        frag_folder = self.config.out_dir
        graph_metrics = []
        for bam, vcf, chromosome in path_list:
            # precompute graphs and get their feature distributions
            config.bam = bam
            config.vcf = vcf
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
                if 'n_nodes' not in metrics:
                    metrics['n_nodes'] = component.n_nodes
                graph_metrics.append(metrics)
            logging.info("Finished building graphs for %s %s" % (config.bam, chromosome))
        return graph_metrics

    def sample_graphs(self, dataset, num_samples, dataset_type, group):
        if not num_samples:
            num_samples = dataset.shape[0]
        sample_param = {'n': min(num_samples, dataset.shape[0])}
        if isinstance(num_samples, float) and num_samples <= 1:
            sample_param = {'frac': num_samples}
        selected_graphs = dataset.sample(**sample_param, random_state=self.config.seed)
        remaining_graphs = dataset[~dataset.component_path.isin(selected_graphs.component_path)]
        selected_graphs["group"] = group
        if num_samples > dataset.shape[0]:
            logging.info(f'WARNING: {group} not enough graphs to build the {dataset_type} set, kept {dataset.shape[0]}.')
        logging.info("%s %s Dataset contains %d graphs." % (group, dataset_type, selected_graphs.shape[0]))
        return selected_graphs, remaining_graphs

    def generate_training_validation_datasets(self):
        """
        generate dataframe containing path of each graph (saved as a FragGraph object),
         and statistics about the graph such as number of nodes, number of edges, density etc.
        """
        assert os.path.exists(self.panel)
        panel = open(self.panel, 'r').readlines()
        # Generate tuples of BAN, VCF paths and chromosome names for parallelization
        path_list = np.array([[panel[i].strip().split()[0], panel[i].strip().split()[1], self.config.chr_names[j]]
                               for j in range(len(self.config.chr_names)) for i in range(len(panel))])
        path_chunks = np.array_split(path_list, self.config.n_procs)
        logging.info("Running on %d processes" % self.config.n_procs)
        logging.info("Chromosomes/process partition: " + str([np.array2string(chk) for chk in path_chunks]))
        generated_graphs = Parallel(n_jobs=self.config.n_procs)(
            delayed(self.generate_graphs)(path_chunks[i]) for i in range(self.config.n_procs))
        graph_index = pd.DataFrame(list(itertools.chain.from_iterable(generated_graphs)))
        logging.info("Generated %d graphs." % graph_index.shape[0])

        logging.info("Starting Graph Selection")
        if self.selection_config:
            if self.config.size:
                logging.info('WARNING: a selection_config was provided, size will be ignored.')
            training_dataset, validation_dataset = self.select_graphs(graph_index)
            logging.info("Training Dataset contains %d graphs." % (training_dataset.shape[0]))
            logging.info("Validation Dataset contains %d graphs." % (validation_dataset.shape[0]))
        else:
            # Limit the size of the dataset
            selected_graphs, _ = self.sample_graphs(graph_index, self.config.size, 'validation', 'Global')
            # Randomly sample the validation graphs according to the validation_ratio
            validation_dataset, remaining_graphs = self.sample_graphs(selected_graphs, self.config.validation_ratio,
                                                                      'validation', 'Global')
            # The remaining graphs are selected for training
            training_dataset, _ = self.sample_graphs(remaining_graphs, None, 'train', 'Global')
        training_dataset.to_pickle(self.config.experiment_dir + '/train.graph_index')
        validation_dataset.to_pickle(self.config.experiment_dir + '/validate.graph_index')

        logging.info("Finished Building Datasets")
