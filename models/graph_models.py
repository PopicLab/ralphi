import torch as th
import torch.nn as nn
import dgl
import inspect
from torch.nn.functional import elu
from dgl.nn.pytorch.conv import GATv2Conv, GINEConv, GINConv, PNAConv, GCN2Conv, GatedGraphConv, GraphConv


class Embedding(nn.Module):
    """
    Parameters
    ----------
    self.node_features_dim : int
        Number of input node features.
    self.hidden_dim : list of int
        ``hidden_dim[i]`` gives the size of node representations after the i-th NF layer.
        ``len(self.hidden_dim)`` equals the number of NF layers.
    attention_layer: list of (str, int, int) or None
        Control the use of an attention layer after every GCN layer.
        If the str is "GRU" then an attention layer will be used with a number of steps given by the first int
        and a number of edge types given by the second int.
        The default is a GRU layer with 2 steps and 1 edge type after each GCN layer.
    """

    def __init__(self, node_features_dim, hidden_dim, attention_layer=None, n_etypes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_features_dim = node_features_dim
        self.gnn_layers = nn.ModuleList()
        n_layers = len(self.hidden_dim)

        if attention_layer is None:
            attention_layer = [["GRU", 2, n_etypes]] * n_layers

        lengths = [len(self.hidden_dim), len(attention_layer)]
        assert len(set(lengths)) == 1, 'Expect the lengths of self.hidden_dim and attention_layer to be the same, ' \
                                       'got {}'.format(lengths)
        for i in range(n_layers):
            self.gnn_layers.append(Embedding(0, []))
            if attention_layer[i][0] == "GRU":
                self.gnn_layers.append(
                    GatedGraphConv(self.hidden_dim[i], self.hidden_dim[i], attention_layer[i][1],
                                   attention_layer[i][2]))

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats, edge_feat=None, edge_weights=None, etypes=None, attention=False):
        """Update node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals self.node_features_dim in initialization
        edge_feat : FloatTensor of shape (E, M1)
            Edge feature with E the number of edges
        etypes : Tensor of shape (E,) or None
            The type of every edge for th GRU layers. Default is a single type of edges
        Returns
        -------
        embedding : list of FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              len(input_feat) in initialization.
        """
        weights = edge_weights if 'edge_weight' in inspect.getfullargspec(self.gnn_layers[0].forward)[0] else edge_feat
        var = 'edge_weight' if 'edge_weight' in inspect.getfullargspec(self.gnn_layers[0].forward)[0] else 'edge_feat'
        var = {var: weights}
        embedding = []
        for gnn in self.gnn_layers:
            if isinstance(gnn, GatedGraphConv):
                feats = gnn(g, feats, etypes)
            else:
                feats = gnn(g, feats, **var)
            embedding.append(feats)
        return embedding


class GCNv2(Embedding):
    """
    Parameters
    ----------
    alpha : list of float or None
        Influence (fraction) of the initial input feature. Default is 0.1
    lambda_ : list of float or None
        decay of the weight matrix. Default is 1.
    bias : list of bool or None, optional
        Add a learnable bias to the output if True. Default is False.
    """

    def __init__(self, node_features_dim, hidden_dim, attention_layer=None, n_etypes=2, activation=None, alpha=None, lambda_=None,
                 bias=None):
        assert sum([out != node_features_dim for out in hidden_dim]) == 0, 'For GCN all the output feats need to have ' \
                                                                'the same dimension as input, ' \
                                                                'got {}'.format([hidden_dim, node_features_dim])
        super(GCNv2, self).__init__(node_features_dim, hidden_dim, attention_layer, n_etypes)

        n_layers = len(self.hidden_dim)
        default_activation = elu
        default_alpha = 0.1
        default_lambda = 1
        if activation is None:
            activation = [default_activation] * n_layers
        if alpha is None:
            alpha = [default_alpha] * n_layers
        if lambda_ is None:
            lambda_ = [default_lambda] * n_layers
        if bias is None:
            bias = [False] * n_layers

        lengths = [len(self.hidden_dim), len(activation), len(alpha), len(lambda_), len(bias)]
        assert len(set(lengths)) == 1, 'Expect the lengths of self.hidden_dim, activation, ' \
                                       'alpha, lambda, and bias to be the same, ' \
                                       'got {}'.format(lengths)

        i = 0
        for idx, gnn in enumerate(self.gnn_layers):
            if isinstance(gnn, Embedding):
                self.gnn_layers[idx] = GCN2Conv(node_features_dim, i + 1, activation=activation[i],
                                                alpha=alpha[i], lambda_=lambda_[i], bias=bias[i])
                node_features_dim = self.hidden_dim[i]
                i += 1

    def forward(self, g, feats, edge_feat=None, edge_weights=None, etypes=None, attention=False):
        g = dgl.add_self_loop(g)
        embedding = []
        init_feats = feats
        for gnn in self.gnn_layers:
            if isinstance(gnn, GatedGraphConv):
                feats = gnn(g, feats, etypes)
            else:
                # edge weights need normalization  (positive)
                # feats = gnn(g, feats, init_feats, edge_weights)
                feats = gnn(g, feats, init_feats)
            embedding.append(feats)
        return embedding


class GCN(Embedding):
    """
    Parameters
    ----------
    bias : list of bool or None, optional
        Add a learnable bias to the output if True. Default is False.
    """

    def __init__(self, node_features_dim, hidden_dim, attention_layer=None, n_etypes=2, activation=None, norm=None, bias=None):
        super(GCN, self).__init__(node_features_dim, hidden_dim, attention_layer, n_etypes)
        n_layers = len(self.hidden_dim)
        default_activation = elu
        default_norm = 'both'
        if activation is None:
            activation = [default_activation] * n_layers
        if norm is None:
            norm = [default_norm] * n_layers
        if bias is None:
            bias = [False] * n_layers

        lengths = [len(self.hidden_dim), len(activation), len(norm), len(bias)]
        assert len(set(lengths)) == 1, 'Expect the lengths of self.hidden_dim, activation, ' \
                                       'norm, and bias to be the same, ' \
                                       'got {}'.format(lengths)

        i = 0
        for idx, gnn in enumerate(self.gnn_layers):
            if isinstance(gnn, Embedding):
                self.gnn_layers[idx] = GraphConv(node_features_dim, hidden_dim[i], activation=activation[i],
                                                 norm=norm[i], bias=bias[i], weight=True)
                node_features_dim = self.hidden_dim[i]
                i += 1


class GIN(Embedding):
    """
    Parameters
    ----------
    apply_func : nn.Module or None
        The neural network to approximate the features.
    init_eps : float or None
        Initial value of epsilon, the influence of the initial embedding.
    learn_eps : list of bool, optional
        If true, epsilon will be learnable.
    """

    def __init__(self, node_features_dim, hidden_dim, attention_layer=None, n_etypes=2, apply_func=None,
                 aggregator_type=None, activation=None, init_eps=0, learn_eps=False):
        super(GIN, self).__init__(node_features_dim, hidden_dim, attention_layer, n_etypes)
        n_layers = len(self.hidden_dim)
        default_aggregator = 'sum'
        default_activation = elu
        default_activation_seq = nn.ELU()
        if aggregator_type is None:
            aggregator_type = [default_aggregator] * n_layers
        if activation is None:
            activation = [default_activation] * n_layers
        if apply_func is None:
            apply_func = []
            for i in range(n_layers):
                apply_func.append(
                    nn.Sequential(nn.Linear(node_features_dim, self.hidden_dim[i]), nn.BatchNorm1d(self.hidden_dim[i]),
                                  default_activation_seq,
                                  nn.Linear(self.hidden_dim[i], self.hidden_dim[i]), default_activation_seq))
                node_features_dim = self.hidden_dim[i]
        lengths = [len(self.hidden_dim), len(apply_func), len(aggregator_type), len(activation)]
        assert len(set(lengths)) == 1, 'Expect the lengths of self.hidden_dim, ' \
                                       ' aggregator_type, activation and ' \
                                       'and apply_func to be the same, ' \
                                       'got {}'.format(lengths)

        i = 0
        for idx, gnn in enumerate(self.gnn_layers):
            if isinstance(gnn, Embedding):
                self.gnn_layers[idx] = GINConv(apply_func=apply_func[i], aggregator_type=aggregator_type[i],
                                               activation=activation[i], init_eps=init_eps, learn_eps=learn_eps)
                i += 1
