import seq.sim as seq
import seq.frags as frags
from seq.var import Variant
import networkx as nx
# from google.cloud import storage
import ntpath
import os
import pickle
import six
from six.moves.urllib.parse import urlsplit
import random
import os, psutil
from collections import defaultdict
import itertools
from itertools import product
import operator
from networkx.algorithms import bipartite
import logging


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
        if compute_trivial:
            self.check_and_set_trivial()

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
        for i, f1 in enumerate(fragments):
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

    def compute_number_of_variants(self):
        vcf_positions = set()
        for frag in self.fragments:
            for block in frag.blocks:
                for var in block.variants:
                    vcf_positions.add(var.vcf_idx)
        return len(vcf_positions)

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
        subg = self.g.subgraph(connected_component).copy()
        subg_frags = [self.fragments[node] for node in subg.nodes]
        node_mapping = {j: i for (i, j) in enumerate(subg.nodes)}
        node_id2hap_id = None
        if self.node_id2hap_id is not None:
            node_id2hap_id = {i: self.node_id2hap_id[j] for (i, j) in enumerate(subg.nodes)}
        subg_relabeled = nx.relabel_nodes(subg, node_mapping, copy=True)
        return FragGraph(subg_relabeled, subg_frags, node_id2hap_id, compute_trivial)

    def connected_components_subgraphs(self, skip_trivial_graphs=False):
        components = nx.connected_components(self.g)
        subgraphs = []
        for count, component in enumerate(components):
            subgraph = self.extract_subgraph(component, compute_trivial=True)
            if subgraph.trivial and skip_trivial_graphs:
                continue
            subgraphs.append(subgraph)
        return subgraphs


class FragGraphGen:
    def __init__(self, frag_panel_file=None, load_components=False, store_components=False,
                 skip_singletons=True, min_graph_size=1, max_graph_size=float('inf'),
                 skip_trivial_graphs=False, compress=False, debug=False, graph_distribution=None):
        self.frag_panel_file = frag_panel_file
        self.load_components = load_components
        self.store_components = store_components
        self.skip_singletons = skip_singletons
        self.min_graph_size = min_graph_size
        self.max_graph_size = max_graph_size
        self.skip_trivial_graphs = skip_trivial_graphs
        self.compress = compress
        self.debug = debug
        self.graph_distribution = graph_distribution
    def __iter__(self):
        # client = storage.Client() #.from_service_account_json('/full/path/to/service-account.json')
        # bucket = client.get_bucket('bucket-id-here')
        if self.graph_distribution is not None:
            print("first originally:", nx.number_of_nodes(self.graph_distribution.combined_connected_components[0].g))
            random.shuffle(self.graph_distribution.combined_connected_components)
            print("first after shuffle:", nx.number_of_nodes(self.graph_distribution.combined_connected_components[0].g))
            for subgraph in self.graph_distribution.combined_connected_components:
                if subgraph.n_nodes < 2 and (self.skip_singletons or subgraph.fragments[0].n_variants < 2):
                    continue
                if not (self.min_graph_size <= subgraph.n_nodes <= self.max_graph_size):
                    continue
                print("Processing subgraph with ", subgraph.n_nodes, " nodes...")
                yield subgraph
        elif self.frag_panel_file is not None:
            with open(self.frag_panel_file, 'r') as panel:
                for frag_file_fname in panel:
                    logging.info("Fragment file: %s" % frag_file_fname)
                    component_file_fname = frag_file_fname.strip() + ".components"
                    if self.load_components and os.path.exists(component_file_fname):
                        with open(component_file_fname, 'rb') as f:
                            connected_components = pickle.load(f)
                    else:
                        fragments = frags.parse_frag_file(frag_file_fname.strip())
                        graph = FragGraph.build(fragments, compute_trivial=False, compress=self.compress)
                        print("Fragment graph with ", graph.n_nodes, " nodes and ", graph.g.number_of_edges(), " edges")
                        print("Finding connected components...")
                        connected_components = graph.connected_components_subgraphs(
                            skip_trivial_graphs=self.skip_trivial_graphs)
                        if self.store_components:
                            with open(component_file_fname, 'wb') as f:
                                pickle.dump(connected_components, f)

                    if not self.debug:
                        # decorrelate connected components, since otherwise we may end up processing connected components in
                        # the order of the corresponding variants which could result in some unwanted correlation
                        # during training between e.g. if there are certain regions of variants with many errors
                        random.shuffle(connected_components)


                    print("Number of connected components: ", len(connected_components))
                    for subgraph in connected_components:
                        if subgraph.n_nodes < 2 and (self.skip_singletons or subgraph.fragments[0].n_variants < 2):
                            continue
                        if not (self.min_graph_size <= subgraph.n_nodes <= self.max_graph_size):
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
