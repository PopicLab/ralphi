import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

msg = fn.copy_src(src='h', out='m')

def edge_msg(edges):
    weighted_node_feat = edges.src['h'] * edges.data['weight'].unsqueeze(1)
    return {'m': weighted_node_feat}


def reduce(nodes):
    accum = torch.cat((torch.mean(nodes.mailbox['m'], 1), torch.max(nodes.mailbox['m'], 1)[0]), dim=1)
    return {'hm': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(3*in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        i = torch.cat((node.data['h'], node.data['hm']), dim=1)
        h = self.linear(i)
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(message_func=edge_msg, reduce_func=reduce)
        g.apply_nodes(func=self.apply_mod)
        g.ndata.pop('hm')
        return g.ndata.pop('h')
