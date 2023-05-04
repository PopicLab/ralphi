import seq.sim as seq
import seq.frags as frags
import networkx as nx
import engine.config as config_utils
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
from seq.vcf_prep import extract_vcf_for_variants
from seq.vcf_prep import construct_vcf_idx_to_record_dict
import wandb

class FragGraph:
    """
    Nodes are read fragments spanning >= 2 variants
    Edges are inserted between fragments that cover the same variants
    """

    def __init__(self, g, fragments, node_id2hap_id=None, compute_trivial=False, compute_properties=False):
        self.g = g
        self.fragments = fragments
        self.n_nodes = g.number_of_nodes()
        self.node_id2hap_id = node_id2hap_id
        self.trivial = None
        self.hap_a_frags = None
        self.hap_b_frags = None
        self.graph_properties = {}
        if compute_trivial:
            self.check_and_set_trivial()
        if compute_properties:
            self.set_graph_properties()

    def set_ground_truth_assignment(self, node_id2hap_id):
        """
        Store the ground truth haplotype assignment available in simulations
        """
        self.node_id2hap_id = node_id2hap_id

    @staticmethod
    def build(fragments, compute_trivial=False, compress=False):
        if compress and fragments:
            # in a compressed graph: nodes are unique fragments, edge weights correspond to the number
            # of all fragment instances
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
            fragments = frags_unique

        frag_graph = nx.Graph()
        print("Constructing fragment graph from %d fragments" % len(fragments))
        for i, f1 in enumerate(tqdm.tqdm(fragments)):
            frag_graph.add_node(i)
            for j in range(i + 1, len(fragments)):
                f2 = fragments[j]
                if f1.vcf_idx_end < f2.vcf_idx_start:
                    # optimization: since we sort the fragments by vcf_idx_start, do not need to consider
                    # later fragments for potential edge existence due to overlap
                    break
                frag_variant_overlap = f1.overlap(f2)
                if len(frag_variant_overlap) == 0:
                    continue
                n_variants = len(frag_variant_overlap)
                n_conflicts = 0
                for variant_pair in frag_variant_overlap:
                    if variant_pair[0].allele != variant_pair[1].allele:
                        n_conflicts += 1
                weight = 1.0 * (-n_variants + 2 * n_conflicts)
                if compress:
                    weight = weight * f1.n_copies * f2.n_copies
                # TODO(viq): differentiate between no overlap and half conflicts
                # Include zero-weight edges for now so that we ensure a variant only belongs to one connected component,
                # as otherwise half-conflicts can result in a variant being split between two connected components.
                # TODO:(Anant): revisit this since these zero-weight edges provide no phasing information
                if weight != 0:
                    frag_graph.add_edge(i, j, weight=weight)

        # setup node features/attributes
        # for now just a binary value indicating whether the node is part of the solution
        for node in frag_graph.nodes:
            frag_graph.nodes[node]['x'] = [0.0]
            frag_graph.nodes[node]['y'] = [fragments[node].n_variants]
        return FragGraph(frag_graph, fragments, compute_trivial=compute_trivial)

    def set_graph_properties(self):
        # TODO: which properties do we want to save here; probably not diameter since expensive to compute
        self.graph_properties["pos_edges"] = 0
        self.graph_properties["neg_edges"] = 0
        self.graph_properties["zero_edges"] = 0
        self.graph_properties["sum_of_pos_edge_weights"] = 0
        self.graph_properties["sum_of_neg_edge_weights"] = 0
        for _, _, a in self.g.edges(data=True):
            edge_weight = a['weight']
            if edge_weight > 0:
                self.graph_properties["pos_edges"] += 1
                self.graph_properties["sum_of_pos_edge_weights"] += edge_weight
            elif edge_weight < 0:
                self.graph_properties["neg_edges"] += 1
                self.graph_properties["sum_of_neg_edge_weights"] += edge_weight
            else:
                self.graph_properties["zero_edges"] += 1
        degrees = [val for (node, val) in self.g.degree()]
        self.graph_properties["max_degree"] = max(degrees)
        self.graph_properties["min_degree"] = min(degrees)
        self.graph_properties["n_nodes"] = self.g.number_of_nodes()
        self.graph_properties["n_edges"] = self.g.number_of_edges()
        self.graph_properties["density"] = nx.density(self.g)
        self.graph_properties["articulation_points"] = len(list(nx.articulation_points(self.g)))
        self.graph_properties["node_connectivity"] = nx.node_connectivity(self.g)
        self.graph_properties["edge_connectivity"] = nx.edge_connectivity(self.g)
        self.graph_properties["diameter"] = nx.diameter(self.g)
        _, self.graph_properties["num_variants"] = self.get_variants_set()
        self.graph_properties["trivial"] = self.trivial

    def log_graph_properties(self, episode_id):
        for key, value in self.graph_properties.items():
            wandb.log({"Episode": episode_id, "Training: " + key: value})

    def get_variants_set(self):
        vcf_positions = set()
        for frag in self.fragments:
            for block in frag.blocks:
                for var in block.variants:
                    vcf_positions.add(var.vcf_idx)
        return vcf_positions, len(vcf_positions)

    def construct_vcf_for_frag_graph(self, input_vcf, output_vcf, vcf_dict):
        print("updating graph indexes to reflect seperated VCF")
        vcf_positions, _ = self.get_variants_set()
        node_mapping = {j: i for (i, j) in enumerate(sorted(vcf_positions))}

        for frag in self.fragments:
            frag.vcf_idx_start = node_mapping[frag.vcf_idx_start]
            frag.vcf_idx_end = node_mapping[frag.vcf_idx_end]
            for block in frag.blocks:
                block.vcf_idx_start = node_mapping[block.vcf_idx_start]
                block.vcf_idx_end = node_mapping[block.vcf_idx_end]
                for var in block.variants:
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

    def extract_subgraph(self, connected_component, compute_trivial=False, compute_properties=False):
        subg = self.g.subgraph(connected_component).copy()
        subg_frags = [self.fragments[node] for node in subg.nodes]
        node_mapping = {j: i for (i, j) in enumerate(subg.nodes)}
        node_id2hap_id = None
        if self.node_id2hap_id is not None:
            node_id2hap_id = {i: self.node_id2hap_id[j] for (i, j) in enumerate(subg.nodes)}
        subg_relabeled = nx.relabel_nodes(subg, node_mapping, copy=True)
        return FragGraph(subg_relabeled, subg_frags, node_id2hap_id, compute_trivial, compute_properties)

    def connected_components_subgraphs(self, skip_trivial_graphs=False, compute_properties=False):
        components = nx.connected_components(self.g)
        print("Found connected components, now constructing subgraphs...")
        subgraphs = []
        for component in tqdm.tqdm(components):
            subgraph = self.extract_subgraph(component, compute_trivial=True, compute_properties=compute_properties)
            if subgraph.trivial and skip_trivial_graphs:
                continue
            subgraphs.append(subgraph)
        return subgraphs

def load_connected_components(frag_file_fname, load_components=False, store_components=False, compress=False, skip_trivial_graphs=False, compute_properties=False):
    logging.info("Fragment file: %s" % frag_file_fname)
    component_file_fname = frag_file_fname.strip() + ".components"
    if load_components and os.path.exists(component_file_fname):
        with open(component_file_fname, 'rb') as f:
            connected_components = pickle.load(f)
    else:
        fragments = frags.parse_frag_file(frag_file_fname.strip())
        graph = FragGraph.build(fragments, compute_trivial=False, compress=compress)
        print("Fragment graph with ", graph.n_nodes, " nodes and ", graph.g.number_of_edges(), " edges")
        print("Finding connected components...")
        connected_components = graph.connected_components_subgraphs(
            skip_trivial_graphs=skip_trivial_graphs, compute_properties=compute_properties)
        if store_components:
            with open(component_file_fname, 'wb') as f:
                pickle.dump(connected_components, f)
    return connected_components

class FragGraphGen:
    def __init__(self, config, graph_dataset=None):
        self.config = config
        self.graph_dataset = graph_dataset

    def is_invalid_subgraph(self, subgraph):
        return subgraph.n_nodes < 2 and (self.config.skip_singleton_graphs or subgraph.fragments[0].n_variants < 2)

    def is_not_in_size_range(self, subgraph):
        return not (self.config.min_graph_size <= subgraph.n_nodes <= self.config.max_graph_size)

    def __iter__(self):
        # client = storage.Client() #.from_service_account_json('/full/path/to/service-account.json')
        # bucket = client.get_bucket('bucket-id-here')

        if self.graph_dataset is not None:
            index_df = self.graph_dataset
            for index, component_row in index_df.iterrows():
                with open(component_row.component_path, 'rb') as f:
                    if not (self.config.min_graph_size <= component_row["n_nodes"] <= self.config.max_graph_size):
                        continue
                    subgraph = pickle.load(f)[component_row['index']]  # index in to list of graphs
                    if self.is_invalid_subgraph(subgraph):
                        continue
                    print("Processing subgraph with ", subgraph.n_nodes, " nodes...")
                    yield subgraph
            yield None
        elif self.config.frag_panel_file is not None:
            with open(self.config.frag_panel_file, 'r') as panel:
                for frag_file_fname in panel:
                    connected_components = load_connected_components(frag_file_fname,
                                                                     load_components=self.config.load_components,
                                                                     store_components=self.config.store_components,
                                                                     compress=self.config.compress,
                                                                     skip_trivial_graphs=self.config.skip_trivial_graphs,
                                                                     compute_properties=False)
                    if not self.config.debug:
                        # decorrelate connected components, since otherwise we may end up processing connected components in
                        # the order of the corresponding variants which could result in some unwanted correlation
                        # during training between e.g. if there are certain regions of variants with many errors
                        random.shuffle(connected_components)
                    print("Number of connected components: ", len(connected_components))
                    for subgraph in connected_components:
                        if self.is_invalid_subgraph(subgraph) or self.is_not_in_size_range(subgraph):
                            continue
                        print("Processing subgraph with ", subgraph.n_nodes, " nodes...")
                        yield subgraph
                    print("Finished processing file: ", frag_file_fname)
            yield None
        else:
            while True:
                graph = generate_rand_frag_graph()
                for subgraph in graph.connected_components_subgraphs():
                    yield subgraph

class GraphDataset:
    def __init__(self, config, ordering_config=None, validation_mode=False):
        self.config = config
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
            df = self.extract_examples(df, filter_condition, self.ordering_config.global_ranges[filter_condition]["min"],  self.ordering_config.global_ranges[filter_condition]["max"])
        return df

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
            quantiles_lookup = {}
            for rule in group_dict["rules"]:
                rule_dict = group_dict["rules"][rule]
                if "quantiles" in rule_dict:
                    quantiles = rule_dict["quantiles"]
                    buckets = df[rule].quantile(quantiles)
                    quantiles_lookup[rule] = buckets
            for rule in group_dict["rules"]:
                rule_dict = group_dict["rules"][rule]
                if ("min" in rule_dict) and ("max" in rule_dict):
                    subsampled_df = self.extract_examples(subsampled_df, rule, rule_dict["min"], rule_dict["max"])
                elif "quantiles" in rule_dict:
                    subsampled_df = self.extract_examples(subsampled_df, rule, quantiles_lookup[rule][rule_dict["quantiles"][0]], quantiles_lookup[rule][rule_dict["quantiles"][1]])
            subsampled_df = subsampled_df.sample(n=num_samples, random_state=self.ordering_config.seed)
            subsampled_df["group"] = group
            df_combined.append(subsampled_df)
            print("subsampled from group: ", group, subsampled_df, subsampled_df.describe())

        df_single_epoch = pd.concat(df_combined)
        if self.ordering_config.shuffle:
            df_single_epoch = df_single_epoch.sample(frac=1, random_state=self.ordering_config.seed)

        if self.ordering_config.drop_redundant:
            df_single_epoch.drop_duplicates(inplace=True)

        df_epochs = []
        for i in range(self.ordering_config.epochs):
            df_iter = df_single_epoch.copy()
            df_epochs.append(df_iter)

        final_df = pd.concat(df_epochs)

        if self.ordering_config.save_indexes_path is not None:
            final_df.to_pickle(self.ordering_config.save_indexes_path)

        return final_df

    def load_indices(self):
        if os.path.exists(self.fragment_files_panel.strip() + ".index_per_graph"):
            graph_dataset = pd.read_pickle(self.fragment_files_panel.strip() + ".index_per_graph")
        else:
            graph_dataset = pd.DataFrame(self.combined_graph_indexes, columns=self.column_names)
            graph_dataset.to_pickle(self.fragment_files_panel.strip() + ".index_per_graph")
        if self.ordering_config:
            graph_dataset = self.dataset_nested_design(graph_dataset)
        print("graph dataset... ", graph_dataset.describe())
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
            connected_components = load_connected_components(panel[i],
                                                             load_components=self.config.load_components,
                                                             store_components=self.config.store_components,
                                                             compress=self.config.compress,
                                                             skip_trivial_graphs=self.config.skip_trivial_graphs,
                                                             compute_properties=True)

            component_index_combined = []
            if self.vcf_panel is not None:
                vcf_dict = construct_vcf_idx_to_record_dict(vcf_panel[i].strip())
            for component_index, component in enumerate(connected_components):
                component_path = panel[i].strip() + ".components" # + "_" + str(component_index)
                if self.vcf_panel is not None:
                    # in this case, we need graph/vcf files per every single graph for validation
                    component_path = component_path + "_" + str(component_index)
                    if not os.path.exists(component_path + ".vcf"):
                        component.construct_vcf_for_frag_graph(vcf_panel[i].strip(),
                                                                        component_path, vcf_dict)
                metrics = component.graph_properties
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
