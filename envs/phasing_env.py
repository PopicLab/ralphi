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
        if self.frag_graph.trivial:
            return
        self.g = dgl.from_networkx(frag_graph.g.to_directed(), edge_attrs=edge_attrs, node_attrs=features)
        self.g = self.g.to(device) 
        self.assigned = torch.zeros(self.num_nodes * 2)
        self.explorable = torch.zeros(self.num_nodes * 2)
        self.H0 = []
        self.H1 = []
        self.best_reward = 0
        self.current_total_reward = 0
        print("Number of nodes: ", self.frag_graph.n_nodes, ", number of edges: ", self.frag_graph.g.number_of_edges())


class PhasingEnv(gym.Env):
    """
    Genome phasing environment
    """

    def __init__(self, config, record_solutions=False, graph_dataset=None, preloaded_graphs=None):
        self.features = list(feature for feature_name in config.features
                             for feature in constants.FEATURES_DICT[feature_name])
        super(PhasingEnv, self).__init__()
        self.config = config
        if preloaded_graphs:
            preloaded_graphs.set_graph_properties(self.features, config=self.config)
            preloaded_graphs.normalize_edges(self.config.weight_norm, self.config.fragment_norm)
            self.state = State(preloaded_graphs, self.features, config.device)
        else:
            self.graph_gen = iter(graphs.FragGraphGen(config, graph_dataset=graph_dataset))
            self.state = self.init_state()
        if not self.has_state():
            raise ValueError("Environment state was not initialized: no valid input graphs")
        # action space consists of the set of nodes we can assign and a termination step
        self.num_actions = self.state.num_nodes * 2
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = {}
        # other bookkeeping
        self.record = record_solutions
        self.solutions = []

    def init_state(self):
        g = next(self.graph_gen)
        if g is not None:
            g.normalize_edges(self.config.weight_norm, self.config.fragment_norm)
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
            # Assignment node i to Haplotype 1 is encoded as the action i + num nodes
            cut_with_h1 = False
            complement_action = action - self.state.num_nodes
        for nbr in self.state.frag_graph.g.neighbors(complement_action):
            if nbr in hap_list[cut_with_h1]:
                self.state.current_total_reward += self.state.frag_graph.g[complement_action][nbr]['weight']
        if self.state.current_total_reward > self.state.best_reward:
            self.state.best_reward = self.state.current_total_reward
        # Compute the corresponding reward, if clip, the reward ios clipped and normalized
        if not self.config.clip:
            return (self.state.current_total_reward - previous_reward) / norm_factor
        else:
            reward = max((self.state.current_total_reward - self.state.best_reward) / self.state.frag_graph.graph_properties[
                "total_num_frag"], 0)
            return reward

    def is_termination_action(self, action):
        return action == self.state.num_nodes

    def is_out_of_moves(self):
        return len(self.state.H0) + len(self.state.H1) >= self.state.num_nodes

    def has_state(self):
        return self.state is not None

    def update_features(self, hap, action):
        self.state.g.ndata['cut_member_' + hap][action] = 1.0
        features = list(self.state.g.ndata.keys())
        if 'reachability_hap0' in features:
            for node in self.state.frag_graph.graph_properties['compo'][
                self.state.frag_graph.graph_properties['neg_connectivity'][action]]:
                self.state.g.ndata['reachability_' + hap][node] += 1.0 / self.state.num_nodes
        if 'shortest_pos_path_' + hap in features:
            paths = self.state.frag_graph.graph_properties['pos_paths'][action]
            for node in paths:
                if node != action:
                    if self.state.frag_graph.g.nodes[node]['val_pos_path_' + hap] == 0 or \
                            self.state.frag_graph.g.nodes[node][
                                'val_pos_path_' + hap] > paths[node]:
                        self.state.frag_graph.g.nodes[node]['val_pos_path_' + hap] = paths[node]
                        self.state.frag_graph.g.nodes[node]['shortest_pos_path_' + hap] = [paths[node] % 2.]
        if 'edge_homophily' in features:
            for node in self.state.frag_graph.g.neighbors(action):
                homo = 0
                if self.state.g.ndata['cut_member_' + hap][action] == self.state.g.ndata['cut_member_' + hap][node]:
                    homo = 1
                self.state.frag_graph.g.nodes[node]['edge_homophily'][0] = self.state.frag_graph.g.nodes[node]['edge_homophily'][0] + homo / len(self.state.frag_graph.g.neighbors(node))
                self.state.frag_graph.g.nodes[action]['edge_homophily'][0] = self.state.frag_graph.g.nodes[action]['edge_homophily'][0] + homo / len(self.state.frag_graph.g.neighbors(action))
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
            if i in self.state.frag_graph.hap_a_frags:
                frag.assign_haplotype(0.0)
            elif i in self.state.frag_graph.hap_b_frags:
                frag.assign_haplotype(1.0)
            else:
                raise RuntimeError("Fragment wasn't assigned to any cluster")
        self.solutions.append(self.state.frag_graph.fragments)

    def get_solutions(self):
        node_labels = self.state.g.ndata['cut_member_hap0'][:, 0].cpu().numpy().tolist()
        for i, frag in enumerate(self.state.frag_graph.fragments):
            frag.assign_haplotype(node_labels[i])
        return self.state.frag_graph.fragments

    # def get_cut_value(self, node_labels=None):
    #     if not self.config.fragment_norm and not self.config.clip:
    #         return self.state.best_reward
    #     else:
    #         return self.state.best_reward * self.state.frag_graph.graph_properties["total_num_frag"]

    def get_cut_value(self):
        mult = 1
        if self.config.fragment_norm or self.config.clip:
            mult = self.state.frag_graph.graph_properties["total_num_frag"]
        reward = self.state.current_total_reward
        if self.config.take_best:
            reward = self.state.best_reward

        return reward * mult

    def render(self, mode='human'):
        """Display the environment"""
        node_labels = self.state.g.ndata['cut_member_hap0'][:].cpu().squeeze().numpy().tolist()
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
        elif mode == "density":
            vis.visualize_dense_graph(self.state.g.to_networkx(), node_labels, edge_weights)
        else:
            # save the plot to file
            pass

    def get_random_valid_action(self):
        pass

    def get_all_valid_actions(self):
        return (self.state.assigned == 0.).nonzero()

    def get_all_non_neighbour_actions(self):
        return (self.state.explorable == 0.).nonzero()

    def get_all_invalid_actions(self):
        return (self.state.assigned == 1.).nonzero()
