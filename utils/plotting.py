import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot_network(nx_graph, node_labels=None):
    if node_labels is None:
        node_labels = []
    node_positions = nx.spring_layout(nx_graph, iterations=20)
    nx.draw(nx_graph, node_positions, node_color=node_labels, with_labels=True)
    plt.show()

def plot_weighted_network(nx_graph, node_labels=None, edge_labels=None):
    if node_labels is None:
        node_labels = []
    if edge_labels is None:
        edge_labels = []
    pos = nx.spring_layout(nx_graph, iterations=20, k=15, seed=100)
    node_labels = ['green' if x == 1.0 else x for x in node_labels]
    node_labels = ['yellow' if x == 0.0 else x for x in node_labels]
    nx.draw(nx_graph, with_labels=True, node_color=node_labels, edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    # plt.savefig("graph_visualization")
    plt.show()