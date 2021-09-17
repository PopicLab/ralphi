import torch
import gym
from gym import spaces
import dgl
import utils.plotting as vis
import graphs.frag_graph as graphs


class State:
    """
    State consists of the genome fragment graph
    and the nodes assigned to each haplotype (0: H0 1: H1)
    """
    def __init__(self, frag_graph):
        self.frag_graph = frag_graph
        self.g = dgl.DGLGraph()
        edge_attrs = None
        if frag_graph.n_nodes > 1:
            edge_attrs = ['weight']
        self.g.from_networkx(frag_graph.g, edge_attrs=edge_attrs, node_attrs=['x', 'y'])
        self.num_nodes = self.g.number_of_nodes()
        self.assigned = torch.zeros(self.num_nodes + 1)
        self.H1 = []
        print("Number of nodes: ", self.num_nodes, ", number of edges: ", self.g.number_of_edges())


class PhasingEnv(gym.Env):
    """
    Genome phasing environment
    """
    def __init__(self, panel=None, record_solutions=False, skip_singleton_graphs=True):
        super(PhasingEnv, self).__init__()
        self.graph_gen = iter(graphs.FragGraphGen(panel, load_graphs=True, store_graphs=True, load_components=True,
            store_components=True, skip_singletons=skip_singleton_graphs))
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

        norm_factor = 1  # self.state.num_nodes
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
