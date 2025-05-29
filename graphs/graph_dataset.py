import os
import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from copy import deepcopy
import pickle
import itertools
import networkx as nx

from models import constants
from graphs.frag_graph import FragGraph
from seq import frags


class GraphDataset:
    def __init__(self, config, ordering_config=None, validation_mode=False):
        self.config = config
        self.features = list(
            feature for feature_name in config.features for feature in constants.FEATURES_DICT[feature_name])
        self.ordering_config = ordering_config
        # Used when the training and validation sets have to be built from the same dataset
        self.combined_graph_indexes = []
        self.recompute = False
        self.validation_mode = validation_mode
        self.panel = config.panel_train
        if validation_mode:
            self.panel = config.panel_validate
        if ordering_config:
            # Determine the features to compute to perform the data filtering and ordering
            path_panel = self.panel.strip() + ".index_per_graph"
            computed_features = []
            if os.path.exists(path_panel):
                graph_dataset = pd.read_pickle(path_panel)
                computed_features = graph_dataset.columns.tolist()
            selection_features = [feature for group in ordering_config.ordering_ranges for feature in
                                  ordering_config.ordering_ranges[group]["rules"] if feature not in computed_features]
            selection_features += [feature for feature in ordering_config.global_ranges if feature not in computed_features]
            selection_features = list(set(selection_features))
            if selection_features:
                self.features += selection_features
                self.recompute = True
        self.generate_dataframe()

    def extract_examples(self, df, condition, lower_bound, upper_bound):
        return df[(df[condition] >= lower_bound) & (df[condition] <= upper_bound)]

    def round_robin_chunkify(self, df):
        size_ordered = df.sort_values(by=['n_nodes'], ascending=True)
        chunks = []
        for i in range(self.config.n_procs):
            chunks.append(size_ordered.iloc[i:: self.config.n_procs, :])
        return chunks

    def apply_rules(self, df, global_df, rules):
        subsampled_df = df.copy()
        aux_buckets = global_df.copy()
        for rule in rules:
            rule_dict = rules[rule]
            if ("min" in rule_dict) or ("max" in rule_dict):
                min_val = 0
                max_val = 500000
                if "min" in rule_dict:
                    min_val = rule_dict["min"]
                if "max" in rule_dict:
                    max_val = rule_dict["max"]
                subsampled_df = self.extract_examples(subsampled_df, rule, min_val, max_val)
                aux_buckets = self.extract_examples(aux_buckets, rule, min_val, max_val)
            elif "quantiles" in rule_dict:
                quantiles = rule_dict["quantiles"]
                buckets = aux_buckets[rule].quantile(quantiles)
                subsampled_df = self.extract_examples(subsampled_df, rule, buckets[rule_dict["quantiles"][0]],
                                                      buckets[rule_dict["quantiles"][1]])
                aux_buckets = self.extract_examples(aux_buckets, rule, buckets[rule_dict["quantiles"][0]],
                                                    buckets[rule_dict["quantiles"][1]])
        return subsampled_df

    def dataset_nested_design(self, df, ordering_config):
        # parses the nested data_ordering_[train,validation].yaml, which allows arbitrary specifications
        # of training/validation set design for any combination of features as long as they are in the indexing df
        df = self.apply_rules(df, df, ordering_config.global_ranges)

        global_df = df.copy()
        df_combined = []
        for group in ordering_config.ordering_ranges:
            group_dict = ordering_config.ordering_ranges[group]
            subsampled_df = self.apply_rules(df, global_df, group_dict["rules"])
            num_samples = ordering_config.num_samples_per_category_default
            if "num_samples" in group_dict:
                num_samples = group_dict["num_samples"]
            subsampled_df = subsampled_df.sample(n=min(num_samples, subsampled_df.shape[0]), random_state=ordering_config.seed)
            df = df[~df.component_path.isin(subsampled_df.component_path)]
            subsampled_df["group"] = group
            df_combined.append(subsampled_df)

        if df_combined:
            df = pd.concat(df_combined)
        if ordering_config.shuffle:
            df = df.sample(frac=1, random_state=ordering_config.seed)

        df.to_pickle(ordering_config.save_indexes_path)

        return df

    def load_dataframe(self):
        path_panel = self.panel.strip() + ".index_per_graph"
        if self.combined_graph_indexes:
            # Newly constructed dataset
            graph_dataset = pd.DataFrame(self.combined_graph_indexes)
            graph_dataset["group"] = "overall"
            graph_dataset.to_pickle(path_panel)
        elif os.path.exists(path_panel):
            # Loading dataset
            graph_dataset = pd.read_pickle(path_panel)
        else:
            # panel contains a list of paths to datasets to merge in a single dataframe.
            panels = open(self.panel, 'r').readlines()
            datasets = []
            for panel in panels:
                path_sub_panel = panel.strip() + ".index_per_graph"
                datasets.append(pd.read_pickle(path_sub_panel))
            graph_dataset = pd.concat(datasets)
            graph_dataset.to_pickle(path_panel)
        logging.info("graph dataset... %s" % graph_dataset.describe())

        if self.ordering_config is not None:
            # Apply the filters specified by the ordering_config
            graph_dataset = self.dataset_nested_design(graph_dataset, self.ordering_config)

        if self.validation_mode:
            # If it is a validation dataset, creates subsets of graphs for my parallelization purposes
            graph_dataset = self.round_robin_chunkify(graph_dataset)
        return graph_dataset

    def compute_dataset_metrics(self, metrics, frag_graph):
        metric_dict = {}
        for metric in metrics:
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

    def generate_graphs(self, chunk):
        # Get the path of the parent folder of the panel file
        config = deepcopy(self.config)
        logging.root.setLevel(logging.getLevelName(config.logging_level))
        panel_parent = '/'.join(self.panel.strip().split("/")[:-1])
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
                component_path = panel_parent + "/" + config.bam.split('/')[-1] + "_" + chromosome + "_" + str(component_index)
                if not os.path.exists(component_path):
                    with open(component_path, 'wb') as f:
                        pickle.dump(component, f)
                metrics = self.compute_dataset_metrics(config.dataset_metrics, component)
                metrics['component_path'] = component_path
                metrics['chromosome'] = chromosome
                metrics['index'] = component_index
                if 'n_nodes' not in metrics:
                    metrics['n_nodes'] = component.n_nodes
                combined_graph_indexes.append(metrics)
            logging.info("Finished building graphs for %s %s" % (config.bam, chromosome))
        return combined_graph_indexes

    def generate_dataframe(self):
        """
        generate dataframe containing path of each graph (saved as a FragGraph object),
         as well as pre-computed statistics about the graph such as connectivity, size, density etc.
        """
        if os.path.exists(self.panel.strip() + ".index_per_graph") and not self.recompute:
            return
        assert os.path.exists(self.panel)
        panel = open(self.panel, 'r').readlines()
        # Check if the panel combines different panels
        if all([os.path.exists(line.strip() + ".index_per_graph") for line in panel]): return
        chunks = np.array([{'bam_path': panel[i].strip().split()[0], 'vcf_path': panel[i].strip().split()[1], 'chromosome': self.config.chr_names[j]}
                            for j in range(len(self.config.chr_names)) for i in range(len(panel))])
        chunks = np.array_split(chunks, self.config.n_procs)
        logging.info("Running on %d processes" % self.config.n_procs)
        logging.info("Chromosomes/process partition: " + str([np.array2string(chk) for chk in chunks]))
        output = Parallel(n_jobs=self.config.n_procs)(
            delayed(self.generate_graphs)(chunks[i]) for i in range(self.config.n_procs))
        self.combined_graph_indexes = list(itertools.chain.from_iterable(output))
