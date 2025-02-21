import seq.frags as frags
import networkx as nx
from joblib import Parallel, delayed
from copy import deepcopy
import pickle
import os
from collections import defaultdict
import itertools
from itertools import product
import operator
from networkx.algorithms import bipartite
import logging
import tqdm
import pandas as pd

from models import constants
import numpy as np


class FragGraph:
    """
    Nodes are read fragments spanning >= 2 variants
    Edges are inserted between fragments that cover the same variants
    """

    def __init__(self, g, fragments, node_id2hap_id=None, compute_trivial=False):
        self.g = g
        self.fragments = fragments
        self.n_nodes = g.number_of_nodes()
        self.node_id2hap_id = node_id2hap_id
        self.trivial = None
        self.hap_a_frags = None
        self.hap_b_frags = None
        self.graph_properties = {}
        if compute_trivial: self.check_and_set_trivial()

    def set_ground_truth_assignment(self, node_id2hap_id):
        """
        Store the ground truth haplotype assignment available in simulations
        """
        self.node_id2hap_id = node_id2hap_id

    @staticmethod
    def merge_fragments_by_identity(fragments):
        # in a compressed graph: nodes are unique fragments,
        # edge weights are adjusted by the number of all fragment instances
        frags_unique = []  # list of unique fragments, list allows comparisons based on equality only
        search_block_idx = None
        for f in fragments:
            # fragments are sorted by the start index in the vcf
            if search_block_idx is not None and frags_unique[search_block_idx].vcf_idx_start == f.vcf_idx_start:
                try:
                    frags_unique[frags_unique.index(f, search_block_idx)].n_copies += 1
                    continue
                except ValueError:
                    pass
            else:
                search_block_idx = len(frags_unique)
            # new fragment
            frags_unique.append(f)
        return frags_unique

    @staticmethod
    def build(fragments, compute_trivial=False, compress=False):
        logging.debug("Input number of fragments: %d" % len(fragments))
        if compress and fragments:
            logging.debug("Compressing identical fragments")
            fragments = FragGraph.merge_fragments_by_identity(fragments)
            logging.debug("Compressed number of fragments: %d" % len(fragments))
        fragments = frags.split_articulation(fragments)

        frag_graph = nx.Graph()
        logging.debug("Constructing fragment graph from %d fragments" % len(fragments))

        for i, f1 in enumerate(fragments):
            frag_graph.add_node(i)
            for j in range(i + 1, len(fragments)):
                f2 = fragments[j]
                # skip since fragments are sorted by vcf_idx_start
                if f1.vcf_idx_end < f2.vcf_idx_start: break
                frag_variant_overlap = f1.overlap(f2)
                if not frag_variant_overlap: continue
                n_conflicts = 0
                for variant_pair in frag_variant_overlap:
                    if variant_pair[0].allele != variant_pair[1].allele:
                        n_conflicts += 1
                weight = 1.0 * (-len(frag_variant_overlap) + 2 * n_conflicts)
                if compress:
                    weight = weight * f1.n_copies * f2.n_copies
                # TODO(viq): differentiate between no overlap and half conflicts
                if weight != 0: frag_graph.add_edge(i, j, weight=weight)
        return FragGraph(frag_graph, fragments, compute_trivial=compute_trivial)

    def set_graph_properties(self, features, config=None):
        if 'betweenness' in features and "betweenness" not in self.graph_properties:
            k = None
            if config.approximate_betweenness and (config.num_pivots < self.n_nodes):
                # if there is more nodes than num_pivots, use "num_pivots" pivots for betweenness approximation
                k = config.num_pivots
            self.graph_properties["betweenness"] = nx.betweenness_centrality(self.g, k=k, seed=config.seed)

        if 'n_edges' in features and "n_edges" not in self.graph_properties:
            self.graph_properties["n_edges"] = self.g.number_of_edges()

        if 'density' in features and "density" not in self.graph_properties:
            self.graph_properties["density"] = nx.density(self.g)
        self.set_node_features(features)

    def set_node_features(self, features):
        for node in self.g.nodes:
            if 'cut_member_hap0' in features:
                self.g.nodes[node]['cut_member_hap0'] = [0.0]
                self.g.nodes[node]['cut_member_hap1'] = [0.0]

            if 'betweenness' in features:
                self.g.nodes[node]['betweenness'] = [self.graph_properties["betweenness"][node]]

    def check_and_set_trivial(self):
        """
        Checks if the max-cut solution can be trivially computed; if so, the solution is computed and stored.

        In the absence of sequencing errors, we can solve this problem optimally by 2-coloring the bipartite graph
        induced by the connected components computed over the agreement subgraph and conflict edges.
        If any connected component contains an internal conflict edge, the solution is not trivial.
        Algorithm:
         1. let G_neg be the input graph G without positive edges (such that only agreements remain)
         2. let CC be the connected components of G_neg
         3. if a conflict edge exists between a pair of nodes of any component in CC => abort
         3. create a new graph G_cc where each connected component in CC is a node and an edge (i, j)
         is added iff there is at least one positive weight edge in G between any node in i and j
         4. find a 2-way coloring of G_cc -- this provides the split into the two haplotypes
         """

        # check if a conflict edge exists within a connected component of the agreement subgraph
        g_neg, conflict_edges = self.prune_edges_by_sign(operator.gt)
        connected_components = [cc for cc in nx.connected_components(g_neg)]
        for cc in connected_components:
            for u, v in itertools.combinations(cc, 2):
                # check if a conflict edge exists between these two nodes
                if v in conflict_edges[u]:
                    self.trivial = False
                    return

        # check bipartiteness of the graph induced by connected components of the agreement subgraph and conflict edges
        g_cc = nx.Graph()
        for i in range(len(connected_components)):
            g_cc.add_node(i)
            for j in range(i + 1, len(connected_components)):
                # check if a conflict edge exists between these two components
                for u, v in product(connected_components[i], connected_components[j]):
                    if v in conflict_edges[u]:
                        g_cc.add_edge(i, j)
                        break
        self.trivial = bipartite.is_bipartite(g_cc)
        if self.trivial:
            hap_a_partition, hap_b_partition = bipartite.sets(g_cc)
            hap_a = [list(connected_components[i]) for i in hap_a_partition]
            hap_b = [list(connected_components[i]) for i in hap_b_partition]
            self.hap_a_frags = [f for cc in hap_a for f in cc]
            self.hap_b_frags = [f for cc in hap_b for f in cc]

    def prune_edges_by_sign(self, op):
        g = self.g.copy()
        pruned_edges = defaultdict(list)
        for u, v, edge_data in self.g.edges(data=True):
            if op(edge_data['weight'], 0):
                g.remove_edge(u, v)
                pruned_edges[u].append(v)
                pruned_edges[v].append(u)
        return g, pruned_edges

    def extract_subgraph(self, connected_component, compute_trivial=False):
        subg = self.g.subgraph(connected_component)
        subg_frags = [self.fragments[node] for node in subg.nodes]
        node_mapping = {j: i for (i, j) in enumerate(subg.nodes)}
        node_id2hap_id = None
        if self.node_id2hap_id is not None:
            node_id2hap_id = {i: self.node_id2hap_id[j] for (i, j) in enumerate(subg.nodes)}
        subg_relabeled = nx.relabel_nodes(subg, node_mapping, copy=True)
        return FragGraph(subg_relabeled, subg_frags, node_id2hap_id, compute_trivial)

    def connected_components_subgraphs(self, config=None, features=None, skip_trivial_graphs=False):
        components = nx.connected_components(self.g)
        logging.debug("Found connected components, now constructing subgraphs...")
        subgraphs = []
        for component in components:
            subgraph = self.extract_subgraph(component, compute_trivial=True)
            if subgraph.trivial and skip_trivial_graphs: continue
            if features and not subgraph.trivial:
                subgraph.set_graph_properties(features, config=config)
            subgraphs.append(subgraph)
        return subgraphs

def connect_components(components):
    def flip_hap(h): return h if h is None else (h + 1) % 2
    gid2components = {}
    component_map = {i: i for i in range(len(components))}
    for cid, frag_list in enumerate(components):
        for frag in frag_list:
            if frag.fragment_group_id:
                gid2components.setdefault(frag.fragment_group_id, []).append((cid, frag))
    for gid, group in gid2components.items():
        hap = group[0][1].haplotype
        cid_g = component_map[group[0][0]]
        for cid, group_frag in group[1:]:
            if group_frag.haplotype != hap:
                for frag in components[cid]:
                    frag.assign_haplotype(flip_hap(frag.haplotype))
            components[cid_g].extend(components[cid])
            components[cid] = []
            aux = component_map[cid]
            for idx in component_map:
                if component_map[idx] == aux:
                    component_map[idx] = cid_g
    return [c for c in components if len(c)]


class FragGraphGen:
    def __init__(self, config, graph_dataset=None):
        self.config = config
        self.features = list(feature for feature_name in config.features
                             for feature in constants.FEATURES_DICT[feature_name])
        self.graph_dataset = graph_dataset

    def is_invalid_subgraph(self, subgraph):
        return subgraph.n_nodes < 2 and self.config.skip_singleton_graphs

    def __iter__(self):
        if self.config.test_mode:
            fragments = frags.parse_frag_repr(self.config.fragments)
            graph = FragGraph.build(fragments, compress=self.config.compress)
            logging.debug("Built fragment graph with %d nodes and %d edges" % (graph.n_nodes, graph.g.number_of_edges()))
            for subgraph in tqdm.tqdm(graph.connected_components_subgraphs(self.config, self.features)):
                yield subgraph
            yield None
        else:
            for _ in range(self.config.epochs):
                for index, component_row in self.graph_dataset.iterrows():
                    with open(component_row.component_path, 'rb') as f:
                        subgraph = pickle.load(f)
                        if self.is_invalid_subgraph(subgraph): continue
                        logging.debug("Processing subgraph with %d nodes..." % subgraph.n_nodes)
                        if self.features: 
                            subgraph.set_graph_properties(self.features, config=self.config)
                        yield subgraph
            yield None

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
        if not validation_mode:
            self.panel = config.panel
            self.vcf_panel = config.vcf_panel
        else:
            self.panel = config.panel_validation
            self.vcf_panel = config.panel_validation_vcfs
        if ordering_config:
            selection_features = [feature for group in ordering_config.ordering_ranges for feature in
                                  ordering_config.ordering_ranges[group]["rules"] if feature not in self.features + ['n_nodes']]
            selection_features += [feature for feature in ordering_config.global_ranges if feature not in
                                                                                            self.features + ['n_nodes']]
            selection_features = list(set(selection_features))
            if selection_features:
                self.features += selection_features
                self.recompute = True
        self.generate_indices()

    def extract_examples(self, df, condition, lower_bound, upper_bound):
        return df[(df[condition] >= lower_bound) & (df[condition] <= upper_bound)]

    def global_filter(self, df, ordering_config):
        for filter_condition in ordering_config.global_ranges:
            df = self.extract_examples(df, filter_condition,
                                       ordering_config.global_ranges[filter_condition]["min"],
                                       ordering_config.global_ranges[filter_condition]["max"])
        return df

    def round_robin_chunkify(self, df):
        size_ordered = df.sort_values(by=['n_nodes'], ascending=True)
        chunks = []
        for i in range(self.config.n_procs):
            chunks.append(size_ordered.iloc[i:: self.config.n_procs, :])
        return chunks

    def dataset_nested_design(self, df, ordering_config):
        # parses the nested data_ordering_[train,validation].yaml, which allows arbitrary specifications
        # of training/validation set design for any combination of features as long as they are in the indexing df
        df = self.global_filter(df, ordering_config)

        df_combined = []
        for group in ordering_config.ordering_ranges:
            group_dict = ordering_config.ordering_ranges[group]
            subsampled_df = df.copy()
            num_samples = ordering_config.num_samples_per_category_default
            if "num_samples" in group_dict:
                num_samples = group_dict["num_samples"]
            for rule in group_dict["rules"]:
                rule_dict = group_dict["rules"][rule]
                if ("min" in rule_dict) and ("max" in rule_dict):
                    subsampled_df = self.extract_examples(subsampled_df, rule, rule_dict["min"], rule_dict["max"])
                elif "quantiles" in rule_dict:
                    quantiles = rule_dict["quantiles"]
                    buckets = subsampled_df[rule].quantile(quantiles)
                    subsampled_df = self.extract_examples(subsampled_df, rule, buckets[rule_dict["quantiles"][0]],
                                                          buckets[rule_dict["quantiles"][1]]) 
            subsampled_df = subsampled_df.sample(n=min(num_samples, subsampled_df.shape[0]), random_state=ordering_config.seed)
            subsampled_df["group"] = group
            df_combined.append(subsampled_df)

        if df_combined:
            df = pd.concat(df_combined)
        if ordering_config.shuffle:
            df = df.sample(frac=1, random_state=ordering_config.seed)

        if ordering_config.drop_redundant:
            df.drop_duplicates(inplace=True)

        df.to_pickle(ordering_config.save_indexes_path)

        return df

    def load_indices(self):
        path_panel = self.panel.strip() + ".index_per_graph"
        if self.combined_graph_indexes:
            graph_dataset = pd.DataFrame(self.combined_graph_indexes)
            graph_dataset["group"] = "overall"
            graph_dataset.to_pickle(path_panel)
        elif os.path.exists(path_panel):
            graph_dataset = pd.read_pickle(path_panel)
        logging.info("graph dataset... %s" % graph_dataset.describe())

        if self.ordering_config is not None:
            graph_dataset = self.dataset_nested_design(graph_dataset, self.ordering_config)

        if self.validation_mode:
            graph_dataset = self.round_robin_chunkify(graph_dataset)
        return graph_dataset

    def generate_graphs(self, chunk):
        # Get the path of the parent folder of the panel file
        config = deepcopy(self.config)
        logging.root.setLevel(logging.getLevelName(config.logging_level))
        panel_parent = '/'.join(self.panel.strip().split("/")[:-1])
        combined_graph_indexes = []
        for i in range(len(chunk)):
            # precompute graphs and get their feature distributions
            config.bam = chunk[i]['panel']
            config.vcf = chunk[i]['vcf_panel']
            chromosome = chunk[i]['chromosome']
            if chromosome == 'chr20' and config.drop_chr20: continue
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
                rem_list = ['betweenness']
                metrics = dict([(key, val) for key, val in component.graph_properties.items() if key not in rem_list])
                metrics['component_path'] = component_path
                metrics['index'] = component_index
                if 'n_nodes' not in metrics:
                    metrics['n_nodes'] = component.n_nodes
                combined_graph_indexes.append(metrics)
            logging.info("Finished building graphs for %s %s" % (config.bam, chromosome))
        return combined_graph_indexes

    def generate_indices(self):
        """
        generate dataframe containing path of each graph (saved as a FragGraph object),
         as well as pre-computed statistics about the graph such as connectivity, size, density etc.
        """
        if os.path.exists(self.panel.strip() + ".index_per_graph") and not self.recompute:
            return
        assert os.path.exists(self.panel) and os.path.exists(self.vcf_panel)
        panel = open(self.panel, 'r').readlines()
        vcf_panel = open(self.vcf_panel, 'r').readlines()
        assert len(panel) == len(vcf_panel)
        chunks = np.array([{'panel': panel[i].strip(), 'vcf_panel': vcf_panel[i].strip(), 'chromosome': self.config.chr_names[j]}
                            for j in range(len(self.config.chr_names)) for i in range(len(panel))])
        chunks = np.array_split(chunks, self.config.n_procs)
        logging.info("Running on %d processes" % self.config.n_procs)
        logging.info("Chromosomes/process partition: " + str([np.array2string(chk) for chk in chunks]))
        output = Parallel(n_jobs=self.config.n_procs)(
            delayed(self.generate_graphs)(chunks[i]) for i in range(self.config.n_procs))
        self.combined_graph_indexes = list(itertools.chain.from_iterable(output))


def eval_assignment_helper(assignments, node2hap):
    correct = 0
    for i in range(len(assignments)):
        true_h = node2hap[i]
        assigned_h = assignments[i]
        if true_h == assigned_h:
            correct += 1
    return 1.0 * correct / len(assignments)


def eval_assignment(assignments, node2hap):
    acc1 = eval_assignment_helper(assignments, node2hap)
    for node in node2hap:
        node2hap[node] = 1 - node2hap[node]
    acc2 = eval_assignment_helper(assignments, node2hap)
    return max(acc1, acc2)
