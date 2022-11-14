import torch
import gym
from gym import spaces
import dgl
import utils.plotting as vis
import graphs.frag_graph as graphs
import networkx as nx
import engine.constants as constants
from collections import defaultdict
from networkx.algorithms import bipartite
from itertools import product
import random
import copy
import operator


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
    def __init__(self, panel=None, record_solutions=False, skip_singleton_graphs=True, min_graph_size=1,
                 max_graph_size=float('inf'), skip_trivial_graphs=False, compress=False, debug=False, graph_distribution=None, preloaded_graphs=None):
        super(PhasingEnv, self).__init__()
        self.graph_gen = iter(graphs.FragGraphGen(panel, load_components=True, store_components=True,
                                                  skip_singletons=skip_singleton_graphs,
                                                  min_graph_size=min_graph_size, max_graph_size=max_graph_size,
                                                  skip_trivial_graphs=skip_trivial_graphs, compress=compress, debug=debug, graph_distribution=graph_distribution))
        self.state = self.init_state()
        if not self.has_state():
            raise ValueError("Environment state was not initialized: no valid input graphs")
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
        norm_factor = 1 #self.state.num_nodes  # 1
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

    def lookup_error_free_instance(self):
        assert self.state.frag_graph.trivial, "Looking up a solution for a non-trivial graph!"
        assert self.state.frag_graph.hap_a_frags is not None and self.state.frag_graph.hap_b_frags is not None, \
            "The solution was not computed"
        for i, frag in enumerate(self.state.frag_graph.fragments):
            if i in self.state.frag_graph.hap_a_frags:
                frag.assign_haplotype(0.0)
            elif i in self.state.frag_graph.hap_b_frags:
                frag.assign_haplotype(1.0)
            else:
                raise RuntimeError("Fragment wasn't assigned to any cluster")
        self.solutions.append(self.state.frag_graph.fragments)

    def get_simple_graph_stats(self):
        return {
            constants.GraphStats.num_nodes: self.state.frag_graph.g.number_of_nodes(),
            constants.GraphStats.num_edges: self.state.frag_graph.g.number_of_edges(),
            constants.GraphStats.density: self.get_density(),
            constants.GraphStats.cut_value: self.get_cut_value()
        }

    def get_graph_stats(self):
        return {
            constants.GraphStats.num_nodes: self.state.frag_graph.g.number_of_nodes(),
            constants.GraphStats.num_edges: self.state.frag_graph.g.number_of_edges(),
            constants.GraphStats.density: self.get_density(),
            constants.GraphStats.radius: self.get_radius(),
            constants.GraphStats.diameter: self.get_diameter(),
            constants.GraphStats.n_variants: self.state.frag_graph.compute_number_of_variants(),
            constants.GraphStats.cut_value: self.get_cut_value()
        }

    def get_density(self):
        return nx.density(self.state.frag_graph.g)

    def get_radius(self):
        return nx.radius(self.state.frag_graph.g)

    def get_diameter(self):
        return nx.diameter(self.state.frag_graph.g)

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
        elif mode == "large":
            vis.visualize_graph(self.state.g.to_networkx(), node_labels, edge_weights)
        else:
            # save the plot to file
            pass

    def get_random_valid_action(self):
        pass

    def get_all_valid_actions(self):
        return (self.state.assigned == 0.).nonzero()

    def get_all_invalid_actions(self):
        return (self.state.assigned == 1.).nonzero()
