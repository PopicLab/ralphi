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
import operator
from networkx.algorithms import bipartite


class FragGraph:
    """
    Nodes are read fragments spanning >= 2 variants
    Edges are inserted between fragments that cover the same variants
    """

    def __init__(self, g, fragments, node_id2hap_id=None, check_and_set_trivial=False):
        self.g = g
        self.fragments = fragments
        self.n_nodes = g.number_of_nodes()
        self.node_id2hap_id = node_id2hap_id
        self.trivial = None
        if check_and_set_trivial:
            self.trivial = self.is_trivial()

    def set_ground_truth_assignment(self, node_id2hap_id):
        """
        Store the ground truth haplotype assignment available in simulations
        """
        self.node_id2hap_id = node_id2hap_id

    @staticmethod
    def build(fragments, check_and_set_trivial=False):
        frag_graph = nx.Graph()
        print("constructing fragment graph")
        for i, f1 in enumerate(fragments):
            frag_graph.add_node(i)
            if i and i % 1000 == 0:
                print("Processing ", i)
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
                # TODO(viq): differentiate between no overlap and half conflicts
                # Include zero-weight edges for now so that we ensure a variant only belongs to one connected component,
                # as otherwise half-conflicts can result in a variant being split between two connected components.
                # TODO:(Anant): revisit this since these zero-weight edges provide no phasing information
                # if weight != 0:
                frag_graph.add_edge(i, j, weight=weight)

        # setup node features/attributes
        # for now just a binary value indicating whether the node is part of the solution
        for node in frag_graph.nodes:
            frag_graph.nodes[node]['x'] = [0.0]
            frag_graph.nodes[node]['y'] = [fragments[node].n_variants]
        return FragGraph(frag_graph, fragments, check_and_set_trivial=check_and_set_trivial)

    def is_trivial(self):
        g_pos, _ = self.prune_edges_by_sign(operator.le)
        return bipartite.is_bipartite(g_pos)

    def prune_edges_by_sign(self, op):
        g = self.g.copy()
        pruned_edges = defaultdict(list)
        for u, v, edge_data in self.g.edges(data=True):
            if op(edge_data['weight'], 0):
                g.remove_edge(u, v)
                pruned_edges[u].append(v)
                pruned_edges[v].append(u)
        return g, pruned_edges

    def extract_subgraph(self, connected_component, check_and_set_trivial=False):
        subg = self.g.subgraph(connected_component).copy()
        subg_frags = [self.fragments[node] for node in subg.nodes]
        node_mapping = {j: i for (i, j) in enumerate(subg.nodes)}
        node_id2hap_id = None
        if self.node_id2hap_id is not None:
            node_id2hap_id = {i: self.node_id2hap_id[j] for (i, j) in enumerate(subg.nodes)}
        subg_relabeled = nx.relabel_nodes(subg, node_mapping, copy=True)
        return FragGraph(subg_relabeled, subg_frags, node_id2hap_id, check_and_set_trivial)

    def connected_components_subgraphs(self, skip_trivial_graphs=False):
        components = nx.connected_components(self.g)
        subgraphs = []
        for count, component in enumerate(components):
            if count and count % 500 == 0:
                print("Processing ", count)
            subgraph = self.extract_subgraph(component, check_and_set_trivial=True)
            if subgraph.trivial and skip_trivial_graphs:
                continue
            subgraphs.append(subgraph)
        return subgraphs


class FragGraphGen:
    def __init__(self, frag_panel_file=None, out_dir=None, load_graphs=False, store_graphs=False, load_components=False,
                 store_components=False, skip_singletons=True, min_graph_size=1, max_graph_size=float('inf'),
                 skip_trivial_graphs=False):
        self.frag_panel_file = frag_panel_file
        self.out_dir = out_dir
        self.load_graphs = load_graphs
        self.store_graphs = store_graphs
        self.load_components = load_components
        self.store_components = store_components
        self.skip_singletons = skip_singletons
        self.min_graph_size = min_graph_size
        self.max_graph_size = max_graph_size
        self.skip_trivial_graphs = skip_trivial_graphs

    def __iter__(self):
        # client = storage.Client() #.from_service_account_json('/full/path/to/service-account.json')
        # bucket = client.get_bucket('bucket-id-here')
        if self.frag_panel_file is not None:
            with open(self.frag_panel_file, 'r') as panel:
                for frag_file_fname in panel:
                    # frag_file_fname = frag_file_fname.replace("\"", "")
                    print("Fragment file: ", frag_file_fname)
                    frag_file_fname_local = frag_file_fname
                    # frag_file_fname_local = '/src/data/train/frags/chr20/' + ntpath.basename(frag_file_fname)
                    # if remote file, download
                    # with open(frag_file_fname_local + ntpath.basename(frag_file_fname), 'w') as frag_file:
                    #    client.download_blob_to_file(frag_file_fname, frag_file)
                    fragments = frags.parse_frag_file(frag_file_fname_local.strip())
                    graph_file_fname = frag_file_fname_local.strip() + ".graph"
                    if self.load_graphs and os.path.exists(graph_file_fname):
                        g = nx.read_gpickle(graph_file_fname)
                        graph = FragGraph(g, fragments)
                    else:
                        graph = FragGraph.build(fragments, check_and_set_trivial=False)
                    if self.store_graphs:
                        nx.write_gpickle(graph.g, graph_file_fname)
                    print("Fragment graph with ", graph.n_nodes, " nodes and ", graph.g.number_of_edges(), " edges")

                    print("Finding connected components...")
                    component_file_fname = frag_file_fname_local.strip() + ".components"
                    if self.load_components and os.path.exists(component_file_fname):
                        with open(component_file_fname, 'rb') as f:
                            connected_components = pickle.load(f)
                    else:
                        connected_components = graph.connected_components_subgraphs(
                            skip_trivial_graphs=self.skip_trivial_graphs)
                        if self.store_components:
                            with open(component_file_fname, 'wb') as f:
                                pickle.dump(connected_components, f)
                    # decorrelate connected components, since otherwise we may end up processing connected components in
                    # the order of the corresponding variants which could result in some unwanted correlation
                    # during training between e.g. if there are certain regions of variants with many errors
                    random.shuffle(connected_components)
                    for subgraph in connected_components:
                        if subgraph.n_nodes < 2 and (self.skip_singletons or subgraph.fragments[0].n_variants < 2):
                            continue
                        if not (self.min_graph_size < subgraph.n_nodes < self.max_graph_size):
                            continue

                        print("Processing subgraph with ", subgraph.n_nodes, " nodes...")
                        yield subgraph
                    print("Finished processing file: ", frag_file_fname)
                    if self.out_dir is not None:
                        with open(self.out_dir + "/training.log", "a") as f:
                            f.write(frag_file_fname)
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
