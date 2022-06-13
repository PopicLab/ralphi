import torch
import gym
from gym import spaces
import dgl
import utils.plotting as vis
import graphs.frag_graph as graphs
import networkx as nx
import random

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
    def __init__(self, panel=None, out_dir=None, record_solutions=False, skip_singleton_graphs=True, prune_graphs_smaller_than=1):
        super(PhasingEnv, self).__init__()
        self.prune_graphs_smaller_than = prune_graphs_smaller_than
        self.graph_gen = iter(graphs.FragGraphGen(panel, out_dir, load_graphs=False, store_graphs=False, load_components=False,
            store_components=False, skip_singletons=skip_singleton_graphs))
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
        while True:
            g = next(self.graph_gen)
            if g is None:
                return None
            if g.n_nodes < self.prune_graphs_smaller_than:
                continue
            return State(g)

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

    def initializeRandomCut(self):
        # helper to experiment with random cuts
        for action in range(self.state.num_nodes):
            flip = random.randint(0, 1)
            if flip == 0:
                self.state.g.ndata['x'][action] = 1.0
                self.state.adjacency_matrix[0][action] = 1.0
                self.state.H1.append(action)

    def getCutValue(self):
        node_labels = self.state.g.ndata['x'][:].cpu().squeeze().numpy().tolist()
        if not isinstance(node_labels, list):
            node_labels = [node_labels]
        S = {i for i, e in enumerate(node_labels) if e != 0}
        netXGraph = self.state.frag_graph.g
        return nx.cut_size(netXGraph, S, weight='weight')

    def render(self, mode='human'):
        """Display the environment"""
        node_labels = self.state.g.ndata['x'][:].cpu().squeeze().numpy().tolist()
        if mode == 'view':
            vis.plot_network(self.state.g.to_networkx(), node_labels)
        else:
            # save the plot to file
            pass

    def get_random_valid_action(self):
        pass

    def get_all_valid_actions(self):
        return (self.state.assigned == 0.).nonzero()

    def get_all_invalid_actions(self):
        return (self.state.assigned == 1.).nonzero()
