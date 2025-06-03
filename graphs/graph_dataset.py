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
    def __init__(self, config):
        self.config = config
        self.features = list(
            feature for feature_name in config.features for feature in constants.FEATURES_DICT[feature_name])

        self.panel = config.panel
        shutil.copyfile(config.panel, config.experiment_dir + '/panel.txt')

        filter_metrics = []
        if config.filter_config:
            self.filter_config = config_utils.load_config(config.filter_config,
                                                          config_type=config_utils.CONFIG_TYPE.DATA_FILTER)
            # Copy the filter used in the local folder
            with open(config.experiment_dir + '/filters.yaml', 'w') as outfile:
                yaml.safe_dump(self.filter_config.__dict__, outfile, default_flow_style=False)

            # Determine the features to compute to perform the data filtering and ordering
            filter_metrics = [feature for group in self.filter_config.filter_categories for feature in
                              self.filter_config.filter_categories[group]["filters"]]
            filter_metrics += [feature for feature in self.filter_config.global_filters]
        self.filter_metrics = list(set(filter_metrics))

    @staticmethod
    def round_robin(n_procs, df):
        size_ordered = df.sort_values(by=['n_nodes'], ascending=True)
        chunks = []
        for i in range(min(n_procs, len(size_ordered))):
            chunks.append(size_ordered.iloc[i:: n_procs, :])
        return chunks

    @staticmethod
    def filter_dataset(df, global_df, filters):
        def extract_examples(extracted_df, metric, lower_bound, upper_bound):
            if lower_bound:
                extracted_df = extracted_df[extracted_df[metric] >= lower_bound]
            if upper_bound:
                extracted_df = extracted_df[extracted_df[metric] <= upper_bound]
            return extracted_df

        subsampled_df = df.copy()
        # Quantiles have to be computed from the original dataset after global filters to be accurate
        aux_buckets = global_df.copy()
        for filter_name, filter_dict in filters.items():
            if ("min" in filter_dict) or ("max" in filter_dict):
                min_val = None
                max_val = None
                if "min" in filter_dict:
                    min_val = filter_dict["min"]
                if "max" in filter_dict:
                    max_val = filter_dict["max"]
                subsampled_df = extract_examples(subsampled_df, filter_name, min_val, max_val)
                aux_buckets = extract_examples(aux_buckets, filter_name, min_val, max_val)
            elif "quantiles" in filter_dict:
                quantiles = filter_dict["quantiles"]
                buckets = aux_buckets[filter_name].quantile(quantiles)
                subsampled_df = extract_examples(subsampled_df, filter_name, buckets[filter_dict["quantiles"][0]],
                                                 buckets[filter_dict["quantiles"][1]])
                aux_buckets = extract_examples(aux_buckets, filter_name, buckets[filter_dict["quantiles"][0]],
                                               buckets[filter_dict["quantiles"][1]])
            else:
                raise(NotImplementedError, 'Please use a filter using min/max or quantiles.')
        return subsampled_df

    def dataset_nested_design(self, df):
        # parses the nested data_ordering_[train,validation].yaml, which allows arbitrary specifications
        # of training/validation set design for any combination of features as long as they are in the indexing df
        df = self.filter_dataset(df, df, self.filter_config.global_filters)
        training_df = []
        validation_df = []
        global_df = df.copy()
        for group in self.filter_config.filter_categories:
            group_dict = self.filter_config.filter_categories[group]
            subsampled_df = self.filter_dataset(df, global_df, group_dict["filters"])
            logging.info("Selected %d graphs for group %s." % (subsampled_df.shape[0], group))
            num_samples_train = self.config.num_samples_train
            if "num_samples_train" in group_dict:
                num_samples_train = group_dict["num_samples_train"]
            num_samples_validate = self.config.num_samples_validate
            if "num_samples_validate" in group_dict:
                num_samples_validate = group_dict["num_samples_validate"]
            # Select the validation graphs and remove them from the available graphs
            val_graphs, subsampled_df = self.sample_datasets(subsampled_df, num_samples_validate, 'validation', group+' ')
            df = df[~df.component_path.isin(val_graphs.component_path)]
            val_graphs["group"] = group
            validation_df.append(val_graphs)

            # Select the training graphs from the remain graphs, if num_samples_train is None, all remaining selected graphs are kept for training
            train_graphs, _ = self.sample_datasets(subsampled_df, num_samples_train, 'train', group+' ')

            df = df[~df.component_path.isin(train_graphs.component_path)]
            train_graphs["group"] = group
            training_df.append(train_graphs)

        training_df = pd.concat(training_df)
        if self.filter_config.shuffle:
            training_df = training_df.sample(frac=1, random_state=self.config.seed)

        validation_df = pd.concat(validation_df)

        return training_df, validation_df

    @staticmethod
    def load_dataframe(panel):
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
                raise ValueError(f'Metric {metric} not defined, please use one of {constants.DATASET_FEATURES}.')
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

    def sample_datasets(self, df, num_samples, dataset_type, group=''):
        if not num_samples:
            num_samples = df.shape[0]
        kept_df = df.sample(n=min(num_samples, df.shape[0]), random_state=self.config.seed)
        subsampled_df = df[~df.component_path.isin(kept_df.component_path)]

        if num_samples > df.shape[0]:
            logging.info(f'WARNING: {group}not enough graphs to build the {dataset_type} set, kept {df.shape[0]}.')
        logging.info("%s%s Dataset contains %d graphs." % (group, dataset_type, kept_df.shape[0]))
        return kept_df, subsampled_df

    def generate_dataframe(self):
        """
        generate dataframe containing path of each graph (saved as a FragGraph object),
         as well as pre-computed statistics about the graph such as connectivity, size, density etc.
        """
        assert os.path.exists(self.panel)
        panel = open(self.panel, 'r').readlines()
        # Check if the panel combines different panels
        chunks = np.array([{'bam_path': panel[i].strip().split()[0], 'vcf_path': panel[i].strip().split()[1],
                            'chromosome': self.config.chr_names[j]}
                           for j in range(len(self.config.chr_names)) for i in range(len(panel))])
        chunks = np.array_split(chunks, self.config.n_procs)
        logging.info("Running on %d processes" % self.config.n_procs)
        logging.info("Chromosomes/process partition: " + str([np.array2string(chk) for chk in chunks]))
        output = Parallel(n_jobs=self.config.n_procs)(
            delayed(self.generate_graphs)(chunks[i]) for i in range(self.config.n_procs))
        combined_graph_indexes = pd.DataFrame(list(itertools.chain.from_iterable(output)))
        logging.info("Generated %d graphs." % combined_graph_indexes.shape[0])

        logging.info("Starting Graph Selection")
        if self.config.filter_config is not None:
            train, val = self.dataset_nested_design(combined_graph_indexes)
            logging.info("Training Dataset contains %d graphs." % (train.shape[0]))
            logging.info("Validation Dataset contains %d graphs." % (val.shape[0]))
        else:
            val, subsampled_df = self.sample_datasets(combined_graph_indexes, self.config.num_samples_validate, 'validation')

            train, _ = self.sample_datasets(subsampled_df, self.config.num_samples_train, 'train')
        train.to_pickle(self.config.experiment_dir + '_train.index_per_graph')
        val.to_pickle(self.config.experiment_dir + '_validate.index_per_graph')

        logging.info("Finished Building DataSets")
