import torch
import gym
from gym import spaces
import dgl
import utils.plotting as vis
import graphs.frag_graph as graphs
import networkx as nx
from collections import defaultdict
from networkx.algorithms import bipartite
from itertools import product
import random
import copy

class State:
    """
    State consists of the genome fragment graph
    and the nodes assigned to each haplotype (0: H0 1: H1)
    """
    def __init__(self, frag_graph):
        self.frag_graph = frag_graph
        edge_attrs = None
        if frag_graph.n_nodes > 1:
            edge_attrs = ['weight']
        self.g = dgl.from_networkx(frag_graph.g.to_directed(), edge_attrs=edge_attrs, node_attrs=['x', 'y'])
        self.num_nodes = self.g.number_of_nodes()
        self.assigned = torch.zeros(self.num_nodes + 1)
        self.H1 = []
        print("Number of nodes: ", self.frag_graph.n_nodes, ", number of edges: ", self.frag_graph.g.number_of_edges())

class PhasingEnv(gym.Env):
    """
    Genome phasing environment
    """
    def __init__(self, panel=None, out_dir=None, record_solutions=False, skip_singleton_graphs=True, min_graph_size=1, max_graph_size=float('inf'), skip_error_free_graphs=False):
        super(PhasingEnv, self).__init__()
        self.graph_gen = iter(graphs.FragGraphGen(panel, out_dir, load_graphs=False, store_graphs=False, load_components=False,
            store_components=False, skip_singletons=skip_singleton_graphs, min_graph_size=min_graph_size, max_graph_size=max_graph_size, skip_error_free_graphs=skip_error_free_graphs))
        self.state = self.init_state()
        # action space consists of the set of nodes we can assign and a termination step
        self.num_actions = self.state.num_nodes + 1
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = {}
        # other bookkeeping
        self.current_total_reward = 0
        self.record = record_solutions
        self.solutions = []

    def init_state(self):
        g = next(self.graph_gen)
        if g is not None:
            return State(g)
        else:
            return None

    def compute_mfc_reward(self, action):
        """The reward is the normalized change in MFC score = sum of all conflict edges from the selected node
        to the remaining graph """

        # experimentally normalizing by number of nodes appears to stablilize actor-critic training
        # (and this normalization seems to be done in literature as well) -- but should confirm this with a side by side comparison
        # TODO (Anant): get a long running result of training with and without normalization
        norm_factor = self.state.num_nodes # 1
        # compute the new MFC score
        previous_reward = self.current_total_reward
        # for each neighbor of the selected node in the graph
        for nbr in self.state.frag_graph.g.neighbors(action):
            if nbr not in self.state.H1:
                self.current_total_reward += self.state.frag_graph.g[action][nbr]['weight']
            else:
                self.current_total_reward -= self.state.frag_graph.g[action][nbr]['weight']
        return (self.current_total_reward - previous_reward) / norm_factor

    def is_termination_action(self, action):
        return action == self.state.num_nodes

    def is_out_of_moves(self):
        return len(self.state.H1) >= self.state.num_nodes

    def has_state(self):
        return self.state is not None

    def step(self, action):
        """Execute one action from given state """
        """Return: next state, reward from current state, is_done, info """

        # assert action is a valid node and it has not been selected yet
        # update the current state by marking the node as selected
        self.state.assigned[action] = 1.0
        if not self.is_termination_action(action):
            self.state.g.ndata['x'][action] = 1.0
            self.state.H1.append(action)
            r_t = self.compute_mfc_reward(action)
            is_done = self.is_out_of_moves()
        else:
            r_t = 0
            is_done = True
        if is_done:
            self.finalize()
        return self.state, r_t, is_done

    def finalize(self):
        if self.record:
            node_labels = self.state.g.ndata['x'][:, 0].cpu().numpy().tolist()
            for i, frag in enumerate(self.state.frag_graph.fragments):
                frag.assign_haplotype(node_labels[i])
            self.solutions.append(self.state.frag_graph.fragments)

    def reset(self):
        """
        Reset the environment to an initial state
        Returns the initial state and the is_done token
        """
        self.state = self.init_state()
        return self.state, not self.has_state()

    def solve_error_free_instance(self):
        """
        In the absence of sequencing errors, we can solve this problem optimally by 2-coloring the bipartite graph
        induced by the connected components computed over negative weight edges only.
        Algorithm:
         1. let G_neg be the input graph G without positive edges (such that only agreements remain)
         2. let CC be the connected components of G_neg
         3. create a new graph G_bipartite where each connected component in CC is a node and an edge (i, j)
         is added iff there is at least one positive weight edge in G between any node in i and j
         4. find a 2-way coloring of G_bipartite -- this provides the split into the two haplotypes
        """
        assert(not self.state.frag_graph.has_seq_error), "Running exact algorithm on a graph with sequencing error!" 
        g_neg, conflict_edges = self.state.frag_graph.extract_negative_edge_subgraph()
        g_bipartite = nx.Graph()
        connected_components = [c for c in nx.connected_components(g_neg)]
        for i in range(len(connected_components)):
            g_bipartite.add_node(i)
            for j in range(i+1, len(connected_components)):
                # check if a conflict edge exists between these two components
                for u, v in product(connected_components[i], connected_components[j]):
                    if v in conflict_edges[u]:
                        g_bipartite.add_edge(i, j)
                        break
        hap_a_partition, hap_b_partition = bipartite.sets(g_bipartite)
        hap_a = [list(connected_components[i]) for i in hap_a_partition]
        hap_b = [list(connected_components[i]) for i in hap_b_partition]
        hap_a_nodes = [node for cc in hap_a for node in cc]
        hap_b_nodes = [node for cc in hap_b for node in cc]
        for i, frag in enumerate(self.state.frag_graph.fragments):
            if i in hap_a_nodes:
                frag.assign_haplotype(0.0)
            elif i in hap_b_nodes:
                frag.assign_haplotype(1.0)
            else:
                raise RuntimeError("Fragment wasn't assigned to any cluster")
        self.solutions.append(self.state.frag_graph.fragments)

    def get_graph_stats(self):
        return self.get_cut_value(), self.state.frag_graph.g.number_of_nodes(), self.state.frag_graph.g.number_of_edges()

    def get_cut_value(self, node_labels=None):
        if not node_labels:
            node_labels = self.state.g.ndata['x'][:].cpu().squeeze().numpy().tolist()
        if not isinstance(node_labels, list):
            node_labels = [node_labels]
        computed_cut = {i for i, e in enumerate(node_labels) if e != 0}
        net_x_graph = self.state.frag_graph.g
        return nx.cut_size(net_x_graph, computed_cut, weight='weight')

    def render(self, mode='human'):
        """Display the environment"""
        node_labels = self.state.g.ndata['x'][:].cpu().squeeze().numpy().tolist()
        edge_weights = self.state.g.edata['weight'].cpu().squeeze().numpy().tolist()
        edges_src = self.state.g.edges()[0].cpu().squeeze().numpy().tolist()
        edges_dst = self.state.g.edges()[1].cpu().squeeze().numpy().tolist()
        edge_indices = zip(edges_src, edges_dst)
        edge_weights = dict(zip(edge_indices, edge_weights))
        if mode == 'view':
            vis.plot_network(self.state.g.to_networkx(), node_labels)
        elif mode == 'weighted_view':
            vis.plot_weighted_network(self.state.g.to_networkx(), node_labels, edge_weights)
        elif mode == "bipartite":
            vis.plot_bipartite_network(self.state.g.to_networkx(), node_labels, edge_weights)
        else:
            # save the plot to file
            pass

    def get_random_valid_action(self):
        pass

    def get_all_valid_actions(self):
        return (self.state.assigned == 0.).nonzero()

    def get_all_invalid_actions(self):
        return (self.state.assigned == 1.).nonzero()
