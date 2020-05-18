import seq.sim as seq
import networkx as nx
import utils.plotting as vis


def get_random_graph(num_nodes, edge_prob=0.5):
    return nx.fast_gnp_random_graph(num_nodes, edge_prob)


class HaplotypeFragGraph:
    def __init__(self, g, node_id2hap_id):
        self.graph_nx = g
        self.node_id2hap_id = node_id2hap_id
        self.num_nodes = g.number_of_nodes()


def num_conflicting_alleles(f1, f2):
    start_overlap = max(f1.start, f2.start)
    end_overlap = min(f1.end, f2.end)
    n_conflicts = 0
    n_alleles = 0
    for i in range(start_overlap, end_overlap + 1, 1):
        n_alleles += 1
        if f1.seq[i - f1.start] != f2.seq[i - f2.start]:
            n_conflicts += 1
    return n_conflicts, n_alleles


def generate_rand_haplotype_graph(h_length=30, n_frags=40):
    h1 = seq.generate_rand_haplotype(h_length)
    h2 = seq.get_complement_haplotype(h1)
    h1_frags = seq.get_n_random_substrings_normal_dist(h1, n_frags)
    h2_frags = seq.get_n_random_substrings_normal_dist(h2, n_frags)

    nodes = h1_frags + h2_frags
    frag_graph = nx.Graph()
    node_id2hap_id = {}
    # insert edges
    for i, f1 in enumerate(nodes):
        frag_graph.add_node(i)
        node_id2hap_id[i] = 0
        if i >= n_frags:
            node_id2hap_id[i] = 1
        for j, f2 in enumerate(nodes):
            if i == j:
                continue
            if f1.overlaps(f2):
                n_conflicts, n_alleles = num_conflicting_alleles(f1, f2)
                weight = 1.0 * (-n_alleles + 2*n_conflicts)
                if weight != 0:
                    frag_graph.add_edge(i, j, weight=weight)

    # setup node features/attributes
    # for now just a binary value indicating whether the node is part of the solution
    for node in frag_graph.nodes:
        frag_graph.nodes[node]['x'] = [0.0]
    return HaplotypeFragGraph(frag_graph, node_id2hap_id)


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

#G = generate_rand_haplotype_graph()
#vis.plot_network(G)

