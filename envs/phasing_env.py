import torch
import gym
from gym import spaces
import dgl
import utils.plotting as vis
import graphs.frag_graph as graphs
import networkx as nx
import models.constants as constants


class State:
    """
    State consists of the genome fragment graph
    and the nodes assigned to each haplotype (0: H0 1: H1)
    """
    def __init__(self, frag_graph, features, device):
        self.frag_graph = frag_graph
        edge_attrs = None
        if frag_graph.n_nodes > 1:
            edge_attrs = ['weight']
        self.num_nodes = self.frag_graph.n_nodes
        if self.frag_graph.trivial: return
        self.g = dgl.from_networkx(frag_graph.g.to_directed(), edge_attrs=edge_attrs, node_attrs=features)
        self.g = self.g.to(device) 
        self.assigned = torch.zeros(self.num_nodes * 2)
        self.H0 = []
        self.H1 = []
        self.best_reward = 0
        self.current_total_reward = 0

class PhasingEnv(gym.Env):
    def __init__(self, config, record_solutions=False, graph_dataset=None):
        self.features = list(feature for feature_name in config.features
                             for feature in constants.FEATURES_DICT[feature_name])
        super(PhasingEnv, self).__init__()
        self.config = config
        self.graph_gen = iter(graphs.FragGraphGen(config, graph_dataset=graph_dataset))
        self.state = self.init_state()
        if not self.has_state():
            raise ValueError("Environment state was not initialized: no valid input graphs")
        # other bookkeeping
        self.record = record_solutions
        self.solutions = []

    def init_state(self):
        g = next(self.graph_gen)
        if g is not None:
            return State(g, self.features, self.config.device)
        else:
            return None

    def compute_reward(self, action):
        """The reward is the normalized change in MFC score = sum of all conflict edges from the selected node
        to the remaining graph """

        if self.config.normalization:
            norm_factor = self.state.num_nodes
        else:
            norm_factor = 1
        # compute the new MFC score
        previous_reward = self.state.current_total_reward
        # for each neighbor of the selected node in the graph
        cut_with_h1 = True
        hap_list = [self.state.H0, self.state.H1]
        complement_action = action
        if action > self.state.num_nodes - 1:
            # qssignment node i to Haplotype 1 is encoded as the action i + num nodes
            cut_with_h1 = False
            complement_action = action - self.state.num_nodes
        for nbr in self.state.frag_graph.g.neighbors(complement_action):
            if nbr in hap_list[cut_with_h1]:
                self.state.current_total_reward += self.state.frag_graph.g[complement_action][nbr]['weight']
        if self.state.current_total_reward > self.state.best_reward:
            self.state.best_reward = self.state.current_total_reward
        # compute the corresponding reward, if clip, the reward ios clipped and normalized
        if not self.config.clip:
            return (self.state.current_total_reward - previous_reward) / norm_factor
        else:
            return max((self.state.current_total_reward - self.state.best_reward) /
                       self.state.frag_graph.graph_properties["total_num_frag"], 0)

    def is_out_of_moves(self):
        return len(self.state.H0) + len(self.state.H1) >= self.state.num_nodes

    def has_state(self):
        return self.state is not None

    def update_features(self, hap, action):
        self.state.g.ndata['cut_member_' + hap][action] = 1.0

    def step(self, action):
        """Execute one action from given state """
        """Return: next state, reward from current state, is_done, info """

        # assert action is a valid node and it has not been selected yet
        # update the current state by marking the node as selected
        self.state.assigned[action] = 1.0
        if action > self.state.num_nodes - 1:
            complement_action = action - self.state.num_nodes
            self.state.H1.append(complement_action)
            self.state.assigned[complement_action] = 1.0
            self.update_features('hap1', complement_action)
        else:
            complement_action = action + self.state.num_nodes
            self.state.H0.append(action)
            self.state.assigned[complement_action] = 1.0
            self.update_features("hap0", action)
        r_t = self.compute_reward(action)
        is_done = False
        if self.is_out_of_moves():
            is_done = True
            self.finalize()
        return self.state, r_t, is_done

    def finalize(self):
        if self.record:
            node_labels = self.state.g.ndata['cut_member_hap0'][:, 0].cpu().numpy().tolist()
            for i, frag in enumerate(self.state.frag_graph.fragments):
                frag.assign_haplotype(node_labels[i])
            self.solutions.append(self.state.frag_graph.fragments)

    def postprocess(self):
            self.solutions = graphs.connect_components(self.solutions)

    def reset(self):
        """
        Reset the environment to an initial state
        Returns the initial state and the is_done token
        """
        self.state = self.init_state()
        return self.state, not self.has_state()

    def process_error_free_instance(self):
        assert self.state.frag_graph.trivial
        assert self.state.frag_graph.hap_a_frags is not None and self.state.frag_graph.hap_b_frags is not None, \
            "The solution was not precomputed"
        for i, frag in enumerate(self.state.frag_graph.fragments):
            if i in self.state.frag_graph.hap_a_frags: frag.assign_haplotype(0.0)
            elif i in self.state.frag_graph.hap_b_frags: frag.assign_haplotype(1.0)
            else: raise RuntimeError("Fragment wasn't assigned to any cluster")
        self.solutions.append(self.state.frag_graph.fragments)

    def get_cut_value(self):
        return self.state.current_total_reward

    def render(self, mode='human'):
        """Display the environment"""
        node_labels = self.state.g.ndata['cut_member_hap0'][:].cpu().squeeze().numpy().tolist()
        if mode == 'view':
            vis.plot_network(self.state.g.to_networkx(), node_labels)
        elif mode == 'weighted_view':
            edge_weights = self.state.g.edata['weight'].cpu().squeeze().numpy().tolist()
            edges_src = self.state.g.edges()[0].cpu().squeeze().numpy().tolist()
            edges_dst = self.state.g.edges()[1].cpu().squeeze().numpy().tolist()
            edge_indices = zip(edges_src, edges_dst)
            edge_weights = dict(zip(edge_indices, edge_weights))
            vis.plot_weighted_network(self.state.g.to_networkx(), node_labels, edge_weights)

    def get_all_invalid_actions(self):
        return (self.state.assigned == 1.).nonzero()
