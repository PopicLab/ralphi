import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('Tkagg') # alternative if 'agg' gives Tkinter related compiler issues

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


def plot_bipartite_network(nx_graph, node_labels=None, edge_labels=None):
    if node_labels is None:
        node_labels = []
    if edge_labels is None:
        edge_labels = []
    # draw bipartite graph
    cut_nodes = [i for i, e in enumerate(node_labels) if e != 0]
    pos = nx.drawing.layout.bipartite_layout(nx_graph, nodes=cut_nodes)
    nx.draw(nx_graph, with_labels=True, node_color=node_labels, edge_cmap=plt.cm.Blues, pos=pos)
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

def visualize_graph(G, color, edge_labels):
    plt.close()
    fig = plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    pos=nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, with_labels=False,
                     node_color=color) #, cmap="autumn")#"Set3")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=7)
    plt.savefig("graph_visualization")
    plt.show()
    return fig