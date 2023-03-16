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

def get_graph_properties(g):
    pos_edges = 0
    neg_edges = 0
    zero_weight_edges = 0
    sum_of_pos_edge_weights = 0
    sum_of_neg_edge_weights = 0
    for u, v, a in g.edges(data=True):
        edge_weight = a['weight']
        if edge_weight > 0:
            pos_edges += 1
            sum_of_pos_edge_weights += edge_weight
        elif edge_weight < 0:
            neg_edges += 1
            sum_of_neg_edge_weights += edge_weight
        else:
            zero_weight_edges += 1
    degrees = [val for (node, val) in g.degree()]
    max_degree = max(degrees)
    min_degree = min(degrees)
    return pos_edges, sum_of_pos_edge_weights, neg_edges, sum_of_neg_edge_weights, min_degree, max_degree

class GraphDistribution:
    def __init__(self, fragment_files_panel, vcf_panel=None, load_components=False, store_components=False, save_indexes=False,
                 compress=False, skip_trivial_graphs=True):
        self.combined_graph_indexes = []
        self.fragment_files_panel = fragment_files_panel
        self.vcf_panel = vcf_panel
        self.compress = compress
        self.skip_trivial_graphs = skip_trivial_graphs
        self.load_components = load_components
        self.store_components = store_components
        self.save_indexes = save_indexes
        self.dataset_indexing()
    def extract_and_save_pandas_index_table(self):
        if os.path.exists(self.fragment_files_panel.strip() + ".index_per_graph"):
            indexing_df = pd.read_pickle(self.fragment_files_panel.strip() + ".index_per_graph")
        else:
            indexing_df = pd.DataFrame(self.combined_graph_indexes, columns=["component_path", "index", "n_nodes", "n_edges", "density", "articulation_points", "node connectivity", "edge_connectivity", "diameter",
                                                        "min_degree", "max_degree", "pos_edges", "neg_edges", "sum_of_pos_edge_weights", "sum_of_neg_edge_weights",
                                                        "trivial"])
            indexing_df.to_pickle(self.fragment_files_panel.strip() + ".index_per_graph")
        return indexing_df

    def load_graph_dataset_indices(self, min = 0, max = float('inf')):
        indexing_df = self.extract_and_save_pandas_index_table()
        print("Loading dataset with distribution... ", indexing_df.describe())
        graph_dataset = indexing_df[(indexing_df.n_nodes > min) & (indexing_df.n_nodes < max)]
        print("filtered dataset... ", graph_dataset.describe())
        return graph_dataset

    def filter_range(self, graph, comparator, min_bound=1, max_bound=float('inf')):
        return min_bound <= comparator(graph) <= max_bound
    def select_graphs_by_range(self, comparator=nx.number_of_nodes, min=1, max=float('inf')):
        return filter(lambda graph: self.filter_range(graph.g, comparator, min, max), self.combined_graph_indexes)
    def select_graphs_by_boolean(self, comparator=nx.has_bridges):
        return filter(lambda graph: comparator(graph.g), self.combined_graph_indexes)

    def dataset_indexing(self):
        """
        generate dataframe containing path of each graph (saved as a FragGraph object),
         as well as pre-computed statistics about the graph such as connectivity, size, density etc.
        """
        if os.path.exists(self.fragment_files_panel.strip() + ".index_per_graph"):
            return pd.read_pickle(self.fragment_files_panel.strip() + ".index_per_graph")
        panel = open(self.fragment_files_panel, 'r')
        if self.vcf_panel is not None:
            vcf_panel = open(self.vcf_panel, 'r')
        else:
            vcf_panel = [None] * len(panel) # placeholder for zip operation
        for frag_file_fname, vcf_file_fname in zip(tqdm.tqdm(panel), vcf_panel):
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
            for i, component in enumerate(tqdm.tqdm(connected_components)):
                specific_component_loc = component_file_fname + "_" + str(i)
                if vcf_file_fname is not None:
                    if not os.path.exists(specific_component_loc + ".vcf"):
                        component.construct_vcf_for_specific_frag_graph(vcf_file_fname.strip(), specific_component_loc + ".vcf")
                        print("saved vcf to: ", specific_component_loc + ".vcf")
                else:
                    if not os.path.exists(specific_component_loc):
                        with open(specific_component_loc, 'wb') as f:
                            pickle.dump(component, f)
                            print("saved graph to: ", specific_component_loc)
                # TODO: which properties do we want to save here; probably not diameter since expensive to compute
                pos_edges, sum_of_pos_edge_weights, neg_edges, sum_of_neg_edge_weights, min_degree, max_degree = get_graph_properties(component.g)
                component_index = [specific_component_loc, i, component.g.number_of_nodes(), component.g.number_of_edges(),
                 nx.density(component.g), len(list(nx.articulation_points(component.g))), nx.node_connectivity(component.g),
                                   nx.edge_connectivity(component.g), nx.diameter(component.g), min_degree, max_degree, pos_edges, neg_edges,
                                   sum_of_pos_edge_weights, sum_of_neg_edge_weights,  component.trivial]
                component_index_combined.append(component_index)
                self.combined_graph_indexes.append(component_index)

            if not os.path.exists(frag_file_fname.strip() + ".index_per_graph") and self.save_indexes:
                indexing_df = pd.DataFrame(component_index_combined,
                                           columns=["component_path", "index", "n_nodes", "n_edges", "density", "articulation_points", "node connectivity", "edge_connectivity", "diameter",
                                                    "min_degree", "max_degree", "pos_edges", "neg_edges", "sum_of_pos_edge_weights", "sum_of_neg_edge_weights",
                                                    "trivial"])
                indexing_df.to_pickle(frag_file_fname.strip() + ".index_per_graph")
        return self.load_graph_dataset_indices()
