from enum import Enum

# Graph stats
class GraphStats(str, Enum):
    num_nodes = "num_nodes"
    num_edges = "num_edges"
    density = "density"
    radius = "radius"
    diameter = "diameter"
    n_variants = "n_variants"
    cut_value = "cut_value"

class LossTypes(str, Enum):
    actor_loss = "actor_loss"
    critic_loss = "critic_loss"
    total_loss = "total_loss"
