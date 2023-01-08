from enum import Enum

# Graph stats
class GraphStats(str, Enum):
    n_nodes = "n_nodes"
    n_edges = "n_edges"
    density = "density"
    radius = "radius"
    diameter = "diameter"
    n_variants = "n_variants"
    cut_value = "cut_value"
    articulation_points = "articulation_points"
    node_connectivity = "node connectivity"
    edge_connectivity = "edge_connectivity"
    min_degree = "min_degree"
    max_degree = "max_degree"
    pos_edges = "pos_edges"
    neg_edges = "neg_edges"
    sum_of_pos_edge_weights = "sum_of_pos_edge_weights"
    sum_of_neg_edge_weights = "sum_of_neg_edge_weights"
    trivial = "trivial"
class LossTypes(str, Enum):
    actor_loss = "actor_loss"
    critic_loss = "critic_loss"
    total_loss = "total_loss"
