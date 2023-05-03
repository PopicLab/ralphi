from enum import Enum

class LossTypes(str, Enum):
    actor_loss = "actor_loss"
    critic_loss = "critic_loss"
    total_loss = "total_loss"

class NodeFeatures(str, Enum):
    """
    defines the node features to use during message passing,
    note all of these are precomputed upon graph construction
    due to caching. This enum enables and disables these features.
    """
    ### MANDATORY FEATURES ###
    cut_member_hap0 = 'cut_member_hap0'
    cut_member_hap1 = 'cut_member_hap1'

    ### optional features ###
    """n_variants = 'n_variants'"""
    """
    pos_neighbors = 'pos_neighbors'
    neg_neighbors = 'neg_neighbors'
    is_articulation = 'is_articulation'"""

    """n_nodes = 'n_nodes'
    num_articulation = 'num_articulation'
    diameter = 'diameter'
    density = 'density'
    max_degree = 'max_degree'
    min_degree = "min_degree"
    n_edges = 'n_edges'
    node_connectivity = "node_connectivity"
    edge_connectivity = "edge_connectivity"
    min_weight = 'min_weight'
    max_weight = 'max_weight'"""

    """max_weight_node = 'max_weight_node'
    min_weight_node = 'min_weight_node'
    num_fragments = 'num_fragments'"""

    """reachability_hap0 = 'reachability_hap0'
    reachability_hap1 = 'reachability_hap1'"""

    """max_num_variant = 'max_num_variant'
    min_num_variant = 'min_num_variant'
    avg_num_variant = 'avg_num_variant'"""
    #compression_factor = 'compression_factor'

    # temporarily disabled, due to simulated data
    #min_qscore = 'min_qscore'
    #max_qscore = 'max_qscore'
    #avg_qscore = 'avg_qscore'
    # temporarily disabled, experiment more with bitmap
    #variant_bitmap = 'variant_bitmap'
