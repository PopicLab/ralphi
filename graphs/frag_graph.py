import seq.sim as seq
import seq.frags as frags
from seq.var import Variant
import networkx as nx
#from google.cloud import storage
import ntpath
import os
import six
from six.moves.urllib.parse import urlsplit


class FragGraph:
    """
    Nodes are read fragments spanning >= 2 variants
    Edges are inserted between fragments that cover the same variants
    """
    def __init__(self, g, fragments, node_id2hap_id=None):
        self.g = g
        self.fragments = fragments
        self.n_nodes = g.number_of_nodes()
        self.node_id2hap_id = node_id2hap_id

    def set_ground_truth_assignment(self, node_id2hap_id):
        """
        Store the ground truth haplotype assignment available in simulations
        """
        self.node_id2hap_id = node_id2hap_id

    @staticmethod
    def build(fragments):
        frag_graph = nx.Graph()
        for i, f1 in enumerate(fragments):
            frag_graph.add_node(i)
            if i % 1000 == 0:
                print("Processing ", i)
            for j in range(i + 1, len(fragments)):
                f2 = fragments[j]
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
                if weight != 0:
                    frag_graph.add_edge(i, j, weight=weight)

        # setup node features/attributes
        # for now just a binary value indicating whether the node is part of the solution
        for node in frag_graph.nodes:
            frag_graph.nodes[node]['x'] = [0.0]
            frag_graph.nodes[node]['y'] = [fragments[node].n_variants]
        return FragGraph(frag_graph, fragments)

    def extract_subgraph(self, connected_component):
        subg = nx.Graph()
        subg.add_nodes_from((n, self.g.nodes[n]) for n in connected_component)
        subg.add_edges_from((n, nbr, w)
                            for n, nbrs in self.g.adj.items() if n in connected_component
                            for nbr, w in nbrs.items() if nbr in connected_component)
        subg.graph.update(self.g.graph)
        subg_frags = [self.fragments[node] for node in subg.nodes]
        node_mapping = {j: i for (i, j) in enumerate(subg.nodes)}
        node_id2hap_id = None
        if self.node_id2hap_id is not None:
            node_id2hap_id = {i: self.node_id2hap_id[j] for (i, j) in enumerate(subg.nodes)}
        subg = nx.relabel_nodes(subg, node_mapping, copy=False)
        return FragGraph(subg, subg_frags, node_id2hap_id)

    def connected_components_subgraphs(self):
        components = nx.connected_components(self.g)
        return [self.extract_subgraph(component) for component in components]


class FragGraphGen:
    def __init__(self, frag_panel_file=None, load_graphs=False, store_graphs=False, skip_singletons=True):
        self.frag_panel_file = frag_panel_file
        self.load_graphs = load_graphs
        self.store_graphs = store_graphs
        self.skip_singletons = skip_singletons

    def __iter__(self):
        #client = storage.Client() #.from_service_account_json('/full/path/to/service-account.json')
        #bucket = client.get_bucket('bucket-id-here')
        if self.frag_panel_file is not None:
            with open(self.frag_panel_file, 'r') as panel:
                for frag_file_fname in panel:
                    #frag_file_fname = frag_file_fname.replace("\"", "")
                    print("Fragment file: ", frag_file_fname)
                    frag_file_fname_local = frag_file_fname
                    #frag_file_fname_local = '/src/data/train/frags/chr20/' + ntpath.basename(frag_file_fname)
                    # if remote file, download
                    #with open(frag_file_fname_local + ntpath.basename(frag_file_fname), 'w') as frag_file:
                    #    client.download_blob_to_file(frag_file_fname, frag_file)
                    fragments = frags.parse_frag_file(frag_file_fname_local.strip())
                    graph_file_fname = frag_file_fname_local.strip() + ".graph"
                    if self.load_graphs and os.path.exists(graph_file_fname):
                        g = nx.read_gpickle(graph_file_fname)
                        graph = FragGraph(g, fragments)
                    else:
                        graph = FragGraph.build(fragments)
                    if self.store_graphs:
                        nx.write_gpickle(graph.g, graph_file_fname)
                    print("Fragment graph with ", graph.n_nodes, " nodes and ", graph.g.number_of_edges(), " edges")
                    print("Finding connected components...")
                    for subgraph in graph.connected_components_subgraphs():
                        if subgraph.n_nodes < 2 and (self.skip_singletons or subgraph.fragments[0].n_variants < 2):
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
    return 1.0*correct/len(assignments)


def eval_assignment(assignments, node2hap):
    acc1 = eval_assignment_helper(assignments, node2hap)
    for node in node2hap:
        node2hap[node] = 1 - node2hap[node]
    acc2 = eval_assignment_helper(assignments, node2hap)
    return max(acc1, acc2)


