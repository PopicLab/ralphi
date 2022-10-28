import argparse
import logging
import pickle
import os
import networkx as nx
import graphs.frag_graph as frag_graph
import seq.frags as frags
import utils.plotting as plt
"""
parser = argparse.ArgumentParser(description='graph data generation haplotype phasing')
parser.add_argument('--panel', help='graph fragment panel file')
parser.add_argument('--out_dir', help='output dir')

args = parser.parse_args()
"""

class TrainingDistribution:
    def __init__(self, fragment_files_panel, load_components=False, store_components=False,
                 compress=False, skip_trivial_graphs=True):
        self.combined_connected_components = []
        self.compress = compress
        self.skip_trivial_graphs = skip_trivial_graphs
        component_file_fname = fragment_files_panel.strip() + ".components"
        if load_components and os.path.exists(component_file_fname):
            self.load_components(component_file_fname)
        else:
            self.compute_components(fragment_files_panel)
            if store_components:
                self.store_components(component_file_fname)

    def load_components(self, component_file_fname):
        with open(component_file_fname, 'rb') as f:
            self.combined_connected_components = pickle.load(f)
    def store_components(self, component_file_fname):
        with open(component_file_fname, 'wb') as f:
            pickle.dump(self.combined_connected_components, f)

    def compute_components(self, fragment_files_panel):
        with open(fragment_files_panel, 'r') as panel:
            for frag_file_fname in panel:
                logging.info("Fragment file: %s" % frag_file_fname)
                fragments = frags.parse_frag_file(frag_file_fname.strip())
                graph = frag_graph.FragGraph.build(fragments, compute_trivial=False, compress=self.compress)
                print("Fragment graph with ", graph.n_nodes, " nodes and ", graph.g.number_of_edges(), " edges")
                print("Finding connected components...")
                connected_components = graph.connected_components_subgraphs(
                    skip_trivial_graphs=self.skip_trivial_graphs)
                self.combined_connected_components.extend(connected_components)
    def filter_range(graph, comparator, min_bound=1, max_bound=float('inf')):
        return min_bound <= comparator(graph) <= max_bound
    def select_graphs_by_range(self, comparator=nx.number_of_nodes, min=1, max=float('inf')):
        return filter(lambda graph: self.filter_range(graph.g, comparator, min, max), self.combined_connected_components)
    def select_graphs_by_boolean(self, comparator=nx.has_bridges):
        return filter(lambda graph: comparator(graph.g), self.combined_connected_components)

"""
dist = TrainingDistribution(args.panel, load_components=True, store_components=True)

# e.g. examples of using the API to get subsets of graphs
size_filtered = dist.select_graphs_by_range(nx.number_of_nodes, min=500, max=float('inf'))
print("sizes: ", len(list(size_filtered)))
density_filtered = dist.select_graphs_by_range(nx.density, min=0.75, max=1)
print("density: ", len(list(density_filtered)))
diameter_filtered = dist.select_graphs_by_range(nx.diameter, min=5, max=float('inf'))
print("diameter: ", len(list(diameter_filtered)))
filtered_by_bridges = dist.select_graphs_by_boolean(comparator=nx.has_bridges)
print("bridges:", len(list(filtered_by_bridges)))
"""