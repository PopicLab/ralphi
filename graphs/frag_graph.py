from copy import deepcopy

import seq.frags as frags
import networkx as nx
import pickle
import random
import os
from collections import defaultdict
import itertools
from itertools import product
import operator
from networkx.algorithms import bipartite
import logging
import tqdm
import pandas as pd
import warnings

from models import constants
from seq.utils import extract_vcf_for_variants, construct_vcf_idx_to_record_dict
import wandb


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
        self.set_node_features(features)

    def log_graph_properties(self, episode_id):
        for key, value in self.graph_properties.items():
            if isinstance(value, set) or isinstance(value, list) or isinstance(value, dict):
                continue
            wandb.log({"Episode": episode_id, "Training: " + key: value})

    def get_variants_set(self):
        vcf_positions = set()
        for frag in self.fragments:
            for var in frag.variants:
                vcf_positions.add(var.vcf_idx)
        return vcf_positions, len(vcf_positions)

    def construct_vcf_for_frag_graph(self, input_vcf, output_vcf, vcf_dict):
        vcf_positions, _ = self.get_variants_set()
        node_mapping = {j: i for (i, j) in enumerate(sorted(vcf_positions))}

        for frag in self.fragments:
            frag.vcf_idx_start = node_mapping[frag.vcf_idx_start]
            frag.vcf_idx_end = node_mapping[frag.vcf_idx_end]
            for var in frag.variants:
                var.vcf_idx = node_mapping[var.vcf_idx]

        if os.path.exists(output_vcf + ".vcf"):
            warnings.warn(output_vcf + ".vcf" + ' already exists!')
            return

        extract_vcf_for_variants(vcf_positions, input_vcf, output_vcf + ".vcf", vcf_dict)

        # also store graph
        if not os.path.exists(output_vcf):
            with open(output_vcf, 'wb') as f:
                pickle.dump(self, f)
        else:
            warnings.warn(output_vcf + ' already exists!')

    def set_node_features(self, features):
        for node in self.g.nodes:
            if 'cut_member_hap0' in features:
                self.g.nodes[node]['cut_member_hap0'] = [0.0]
                self.g.nodes[node]['cut_member_hap1'] = [0.0]

            if 'betweenness' in features:
                self.g.nodes[node]['betweenness'] = [self.graph_properties["betweenness"][node]]

    def normalize_edges(self, weight_norm, fragment_norm):
        if weight_norm:
            dict_weights = {k: v / self.graph_properties["sum_of_pos_edge_weights"] for k, v in
                            nx.get_edge_attributes(self.g, 'weight').items()}
            nx.set_edge_attributes(self.g, dict_weights, 'weight')
        if fragment_norm:
            dict_weights = {k: v / self.graph_properties["total_num_frag"] for k, v in
                            nx.get_edge_attributes(self.g, 'weight').items()}
            nx.set_edge_attributes(self.g, dict_weights, 'weight')

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
        subg_frags = [self.fragments[node] if not self.fragments[node] else
                      deepcopy(self.fragments[node]) for node in subg.nodes]
        node_mapping = {j: i for (i, j) in enumerate(subg.nodes)}
        node_id2hap_id = None
        if self.node_id2hap_id is not None:
            node_id2hap_id = {i: self.node_id2hap_id[j] for (i, j) in enumerate(subg.nodes)}
        subg_relabeled = nx.relabel_nodes(subg, node_mapping, copy=True)
        return FragGraph(subg_relabeled, subg_frags, node_id2hap_id, compute_trivial)

    def get_component(self, p, edge_stack, current):
        component = []
        while edge_stack:
            # Get the biconnected component
            e = edge_stack.pop()
            component += e
            if (p is not None) and (p in e) and (current in e):
                # Got back to the articulation
                break
        return list(set(component))

    def tarjan_algorithm(self): # O(V+E)
        articulation_points = set()
        visited = set()
        # When the nodes have been discovered
        disc = {}
        # Records the index of the earliest discovered neighbor
        bic = {}
        parent = {}
        number_explored = 0
        edge_stack = []
        biconnected_components = []
        for start in self.g.nodes():
            # We might have disconnected components so it is necessary to try other nodes.
            if start in visited:
                continue
            # New connected component
            stack = [(start, None, self.g.neighbors(start))]
            disc[start] = bic[start] = number_explored
            number_explored += 1
            visited.add(start)
            parent[start] = None
            children = 0
            # Iterative DFS
            while stack:
                current, prev, neighbors = stack[-1]
                try:
                    neighbor = next(neighbors)
                    if neighbor not in visited:
                        # neighbor is connected to the current start
                        visited.add(neighbor)
                        parent[neighbor] = current
                        disc[neighbor] = bic[neighbor] = number_explored
                        number_explored += 1
                        stack.append((neighbor, current, self.g.neighbors(neighbor)))
                        edge_stack.append([neighbor, current])
                        if parent[current] is None:
                            children += 1
                    elif neighbor != prev:
                        # If current has been connected to a node discovered before it, they are in the same bic
                        bic[current] = min(bic[current], disc[neighbor])
                except StopIteration:
                    stack.pop()
                    p = parent[current]
                    if p is not None and not p == start:
                        # It is not a start node
                        # If current connected to a node discovered before its parent, the parent is in the same bic
                        bic[p] = min(bic[p], bic[current])
                        if bic[current] >= disc[p]:
                            # The parent wasn't reach over this section of the DFS, it is an articulation point
                            articulation_points.add(p)
                            biconnected_components.append(self.get_component(p, edge_stack, current))
                    else:
                        if children > 1:
                            # Start has neighbors only connected through start
                            articulation_points.add(start)
                            if edge_stack:
                                biconnected_components.append(self.get_component(p, edge_stack, current))
            if edge_stack:
                biconnected_components.append(self.get_component(None, edge_stack, None))
        return articulation_points, biconnected_components

    def get_biconnected_subgraphs(self):
        articulation_points, biconnected_components = self.tarjan_algorithm()
        # If there is a single biconnected component, there is no articulation point.
        if len(biconnected_components) == 1: return biconnected_components
        for articulation in articulation_points:
            # This node will be duplicated when taking the subgraphs, keeps its id for stitching
            self.fragments[articulation].fragment_group_id = articulation
            self.fragments[articulation].number_duplicated += 1
        return biconnected_components


    def connected_components_subgraphs(self, config=None, features=None, skip_trivial_graphs=False):
        components = self.get_biconnected_subgraphs()
        logging.debug("Found connected components, now constructing subgraphs...")
        subgraphs = []

        for component in components:
            subgraph = self.extract_subgraph(component, compute_trivial=True)
            if subgraph.trivial and skip_trivial_graphs: continue
            if features and not subgraph.trivial:
                subgraph.set_graph_properties(features, config=config)
            subgraphs.append(subgraph)
        return subgraphs


def load_connected_components(frag_file_fname, features, config):
    logging.debug("Fragment file: %s" % frag_file_fname)
    component_file_fname = frag_file_fname.strip() + ".components"
    if config.load_components and os.path.exists(component_file_fname):
        with open(component_file_fname, 'rb') as f:
            connected_components = pickle.load(f)
    else:
        fragments = frags.parse_frag_file(frag_file_fname.strip())
        graph = FragGraph.build(fragments, features, compute_trivial=False, compress=config.compress)
        logging.debug("Built fragment graph with %d nodes and %d edges" % (graph.n_nodes, graph.g.number_of_edges()))
        logging.debug("Finding connected components...")
        connected_components = graph.connected_components_subgraphs(
            config, features, skip_trivial_graphs=config.skip_trivial_graphs)
        if config.store_components:
            with open(component_file_fname, 'wb') as f:
                pickle.dump(connected_components, f)
    return connected_components


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
            cid = component_map[cid]
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

    def is_not_in_size_range(self, subgraph):
        return not (self.config.min_graph_size <= subgraph.n_nodes <= self.config.max_graph_size)

    def __iter__(self):
        if self.config.test_mode:
            fragments = frags.parse_frag_repr(self.config.fragments)
            graph = FragGraph.build(fragments, compress=self.config.compress)
            logging.debug("Built fragment graph with %d nodes and %d edges" % (graph.n_nodes, graph.g.number_of_edges()))
            for subgraph in tqdm.tqdm(graph.connected_components_subgraphs(self.config, self.features)):
                yield subgraph
            yield None
        elif self.graph_dataset is not None:
            for _ in range(self.config.epochs):
                for index, component_row in self.graph_dataset.iterrows():
                    with open(component_row.component_path, 'rb') as f:
                        if not (self.config.min_graph_size <= component_row["n_nodes"] <= self.config.max_graph_size):
                            continue
                        subgraph = pickle.load(f)
                        if self.is_invalid_subgraph(subgraph): continue
                        logging.debug("Processing subgraph with %d nodes..." % subgraph.n_nodes)
                        if self.features: 
                            subgraph.set_graph_properties(self.features, config=self.config)
                        yield subgraph
            yield None
        elif self.config.frag_panel_file is not None:
            with open(self.config.frag_panel_file, 'r') as panel:
                for frag_file_fname in panel:
                    connected_components = load_connected_components(frag_file_fname, self.features, self.config)
                    random.seed(self.config.seed)
                    random.shuffle(connected_components)
                    for subgraph in connected_components:
                        if self.is_invalid_subgraph(subgraph) or self.is_not_in_size_range(subgraph):
                            continue
                        logging.debug("Processing subgraph with %d nodes..." % subgraph.n_nodes)
                        yield subgraph
                    logging.info("Finished processing file: %s" % frag_file_fname)
            yield None

class GraphDataset:
    def __init__(self, config, ordering_config=None, validation_mode=False):
        self.config = config
        self.features = list(
            feature for feature_name in config.features for feature in constants.FEATURES_DICT[feature_name])
        self.ordering_config = ordering_config
        self.combined_graph_indexes = []
        self.validation_mode = validation_mode
        if not validation_mode:
            self.fragment_files_panel = config.panel
            self.vcf_panel = None
        else:
            self.fragment_files_panel = config.panel_validation_frags
            self.vcf_panel = config.panel_validation_vcfs
        self.column_names = None
        self.generate_indices()

    def extract_examples(self, df, condition, lower_bound, upper_bound):
        return df[(df[condition] >= lower_bound) & (df[condition] <= upper_bound)]

    def global_filter(self, df):
        for filter_condition in self.ordering_config.global_ranges:
            df = self.extract_examples(df, filter_condition,
                                       self.ordering_config.global_ranges[filter_condition]["min"],
                                       self.ordering_config.global_ranges[filter_condition]["max"])
        return df

    def round_robin_chunkify(self, df):
        size_ordered = df.sort_values(by=['n_nodes'], ascending=True)
        chunks = []
        for i in range(self.config.num_cores_validation):
            chunks.append(size_ordered.iloc[i:: self.config.num_cores_validation, :])
        return chunks

    def dataset_nested_design(self, df):
        # parses the nested data_ordering_[train,validation].yaml, which allows arbitrary specifications
        # of training/validation set design for any combination of features as long as they are in the indexing df

        df = self.global_filter(df)

        df_combined = []

        for group in self.ordering_config.ordering_ranges:
            group_dict = self.ordering_config.ordering_ranges[group]
            subsampled_df = df.copy()
            num_samples = self.ordering_config.num_samples_per_category_default
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
            subsampled_df = subsampled_df.sample(n=min(num_samples, subsampled_df.shape[0]), random_state=self.ordering_config.seed)
            subsampled_df["group"] = group
            df_combined.append(subsampled_df)

        if len(df_combined) == 0:
            df_single_epoch = df
            if "group" not in df_single_epoch:
                df_single_epoch["group"] = "original"
        else:
            df_single_epoch = pd.concat(df_combined)
        if self.ordering_config.shuffle:
            df_single_epoch = df_single_epoch.sample(frac=1, random_state=self.ordering_config.seed)

        if self.ordering_config.drop_redundant:
            df_single_epoch.drop_duplicates(inplace=True)

        final_df = df_single_epoch

        if self.validation_mode:
            final_df = self.round_robin_chunkify(final_df)

        if self.ordering_config.save_indexes_path is not None:
            final_df.to_pickle(self.ordering_config.save_indexes_path)

        return final_df

    def load_indices(self):
        if os.path.exists(self.fragment_files_panel.strip() + ".index_per_graph"):
            graph_dataset = pd.read_pickle(self.fragment_files_panel.strip() + ".index_per_graph")
        else:
            graph_dataset = pd.DataFrame(self.combined_graph_indexes, columns=self.column_names)
            graph_dataset.to_pickle(self.fragment_files_panel.strip() + ".index_per_graph")
        logging.info("graph dataset... %s" % graph_dataset.describe())
        if self.ordering_config:
            graph_dataset = self.dataset_nested_design(graph_dataset)
        return graph_dataset

    def generate_indices(self):
        """
        generate dataframe containing path of each graph (saved as a FragGraph object),
         as well as pre-computed statistics about the graph such as connectivity, size, density etc.
        """
        if os.path.exists(self.fragment_files_panel.strip() + ".index_per_graph"):
            return
        panel = open(self.fragment_files_panel, 'r').readlines()
        if self.vcf_panel is not None:
            vcf_panel = open(self.vcf_panel, 'r').readlines()
        for i in range(len(panel)):
            # precompute graph properties when generating distribution
            connected_components = load_connected_components(panel[i], self.features, self.config)
            component_index_combined = []
            if self.vcf_panel is not None:
                vcf_dict = construct_vcf_idx_to_record_dict(vcf_panel[i].strip())
            for component_index, component in enumerate(connected_components):
                component_path = panel[i].strip() + ".components"  # + "_" + str(component_index)
                if self.vcf_panel is not None:
                    # in this case, we need graph/vcf files per every single graph for validation
                    component_path = component_path + "_" + str(component_index)
                    if not os.path.exists(component_path + ".vcf"):
                        component.construct_vcf_for_frag_graph(vcf_panel[i].strip(),
                                                               component_path, vcf_dict)
                else:
                    if not os.path.exists(component_path):
                        with open(component_path, 'wb') as f:
                            pickle.dump(component, f)

                rem_list = ['neg_connectivity', 'compo', 'pos_paths']
                metrics = dict([(key, val) for key, val in component.graph_properties.items() if key not in rem_list])
                component_index = [component_path, component_index] + list(metrics.values())
                if not self.column_names:
                    self.column_names = ["component_path", "index"] + list(metrics.keys())
                component_index_combined.append(component_index)
                self.combined_graph_indexes.append(component_index)

            if not os.path.exists(panel[i].strip() + ".index_per_graph") and self.config.store_indexes:
                indexing_df = pd.DataFrame(component_index_combined,
                                           columns=self.column_names)
                indexing_df.to_pickle(panel[i].strip() + ".index_per_graph")


def generate_rand_frag_graph(h_length=30, n_frags=40):
    h1 = seq.generate_rand_haplotype(h_length)
    h2 = seq.get_complement_haplotype(h1)
    h1_frags = seq.get_n_random_substrings_normal_dist(h1, n_frags)
    h2_frags = seq.get_n_random_substrings_normal_dist(h2, n_frags)
    fragments = h1_frags + h2_frags
    frag_graph = FragGraph.build(fragments)
    node_id2hap_id = {i: 0 if i < n_frags else 1 for i in range(len(fragments))}
    frag_graph.set_ground_truth_assignment(node_id2hap_id)
    return frag_graph


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
