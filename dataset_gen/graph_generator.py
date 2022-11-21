import argparse
import logging
import pickle
import os
import networkx as nx
import graphs.frag_graph as frag_graph
import seq.frags as frags
import utils.plotting as plt
import tqdm
import pandas as pd

"""
parser = argparse.ArgumentParser(description='graph data generation haplotype phasing')
parser.add_argument('--panel', help='graph fragment panel file')

args = parser.parse_args()
"""

class GraphDistribution:
    def __init__(self, fragment_files_panel, load_components=False, store_components=False, save_indexes=False,
                 compress=False, skip_trivial_graphs=True):
        self.combined_graph_indexes = []
        self.fragment_files_panel = fragment_files_panel
        self.compress = compress
        self.skip_trivial_graphs = skip_trivial_graphs
        self.load_components = load_components
        self.store_components = store_components
        self.save_indexes = save_indexes
        self.compute_components()

    def compute_components(self):
        with open(self.fragment_files_panel, 'r') as panel:
            for frag_file_fname in tqdm.tqdm(panel):
                logging.info("Fragment file: %s" % frag_file_fname)
                component_file_fname = frag_file_fname.strip() + ".components"
                if self.load_components and os.path.exists(component_file_fname):
                    with open(component_file_fname, 'rb') as f:
                        connected_components = pickle.load(f)
                else:
                    fragments = frags.parse_frag_file(frag_file_fname.strip())
                    graph = frag_graph.FragGraph.build(fragments, compute_trivial=False, compress=self.compress)
                    print("Fragment graph with ", graph.n_nodes, " nodes and ", graph.g.number_of_edges(), " edges")
                    print("Finding connected components...")
                    connected_components = graph.connected_components_subgraphs(
                        skip_trivial_graphs=self.skip_trivial_graphs)
                    if self.store_components:
                        with open(component_file_fname, 'wb') as f:
                            pickle.dump(connected_components, f)

                component_index_combined = []
                for i, component in enumerate(connected_components):
                    # TODO: which properties do we want to save here; probably not diameter since expensive to compute
                    component_index = [component_file_fname, i, component.g.number_of_nodes(), component.g.number_of_edges(),
                     nx.density(component.g), component.trivial]
                    component_index_combined.append(component_index)
                    self.combined_graph_indexes.append(component_index)

                if not os.path.exists(frag_file_fname.strip() + ".index") and self.save_indexes:
                    indexing_df = pd.DataFrame(component_index_combined,
                                               columns=["component_path", "index", "n_nodes", "n_edges", "density",
                                                        "trivial"])
                    indexing_df.to_pickle(frag_file_fname.strip() + ".index")
    def extract_and_save_pandas_index_table(self):
        if os.path.exists(self.fragment_files_panel.strip() + ".index"):
            indexing_df = pd.read_pickle(self.fragment_files_panel.strip() + ".index")
        else:
            indexing_df = pd.DataFrame(self.combined_graph_indexes, columns=["component_path", "index", "n_nodes", "n_edges", "density", "trivial"])
            indexing_df.to_pickle(self.fragment_files_panel.strip() + ".index")
        return indexing_df

    def load_graph_dataset_indices(self, min = 0, max = float('inf')):
        indexing_df = self.extract_and_save_pandas_index_table()
        print("Loading dataset with distribution... ", indexing_df.describe())
        graph_dataset = indexing_df[(indexing_df.n_nodes > min) & (indexing_df.n_nodes < max)]
        print("filtered dataset... ", graph_dataset.describe())
        return graph_dataset

    def load_graphs_given_indices(self, index_df):
        loaded_graphs = []
        for index, component_row in index_df.iterrows():
            if not os.path.exists(component_row.component_path):
                exit(1)
            else:
                with open(component_row.component_path, 'rb') as f:
                    print("component row:", component_row)
                    print("component row index:", component_row['index'])
                    component_graph = pickle.load(f)[component_row['index']]
                    # print(component_row[index])
                    print(component_graph)
                    loaded_graphs.append(component_graph)
        return loaded_graphs

    def filter_range(self, graph, comparator, min_bound=1, max_bound=float('inf')):
        return min_bound <= comparator(graph) <= max_bound
    def select_graphs_by_range(self, comparator=nx.number_of_nodes, min=1, max=float('inf')):
        return filter(lambda graph: self.filter_range(graph.g, comparator, min, max), self.combined_graph_indexes)
    def select_graphs_by_boolean(self, comparator=nx.has_bridges):
        return filter(lambda graph: comparator(graph.g), self.combined_graph_indexes)
"""
class TrainingDistribution(GraphDistribution):
    def __init__(self, fragment_files_panel, load_components=False, store_components=False, save_indexes=False,
                 compress=False, skip_trivial_graphs=True):
        super().__init__(fragment_files_panel, load_components, store_components, save_indexes,
                 compress, skip_trivial_graphs)
        
class ValidationDistribution(GraphDistribution):
    def __init__(self, fragment_files_panel, load_components=False, store_components=False, save_indexes=False,
                 compress=False, skip_trivial_graphs=True):
        super().__init__(fragment_files_panel, load_components, store_components, save_indexes,
                 compress, skip_trivial_graphs)
"""

"""
dist = TrainingDistribution(args.panel, load_components=True, store_components=True, save_indexes=True)
graph_dataset_indices = dist.load_graph_dataset_indices()
#dist.load_graphs_given_indices(graph_dataset_indices)
"""

"""
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