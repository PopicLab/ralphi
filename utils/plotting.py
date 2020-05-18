import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')


def plot_network(nx_graph, node_labels=None):
    if node_labels is None:
        node_labels = []
    node_positions = nx.spring_layout(nx_graph, iterations=20)
    #nx.draw(nx_graph, node_positions, node_color=node_labels, with_labels=True)
    nx.draw(nx_graph, node_positions, node_color=node_labels, with_labels=True)
    plt.show()

