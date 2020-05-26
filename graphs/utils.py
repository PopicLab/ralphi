import networkx as nx


def get_random_graph(num_nodes, edge_prob=0.5):
    return nx.fast_gnp_random_graph(num_nodes, edge_prob)