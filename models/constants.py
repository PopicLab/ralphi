from enum import Enum


class LossTypes(str, Enum):
    actor_loss = 'actor_loss'
    critic_loss = 'critic_loss'
    total_loss = 'total_loss'


class NodeFeatures(str, Enum):
    """
    defines the node features to use during message passing,
    note all of these are precomputed upon graph construction
    due to caching. This enum enables and disables these features.
    """
    features = {
        ### MANDATORY FEATURES ###
        'dual': [
            'cut_member_hap0',
            'cut_member_hap1'
        ],

        ### optional features ###
        'variants': [
            'n_variants'],

        'neigh': [
            'pos_neighbors'
            , 'neg_neighbors'
            , 'max_weight_node'
            , 'min_weight_node'
        ],

        'arti': [
            'is_articulation'
        ],

        'stats': [
            'n_nodes'
            , 'num_articulation'
            , 'diameter'
            , 'density'
            , 'max_degree'
            , 'min_degree'
            , 'n_edges'
            , 'node_connectivity'
            , 'edge_connectivity'
            , 'min_weight'
            , 'max_weight'
            , 'num_fragments'
            , 'compression_factor'
            , 'max_num_variant'
            , 'min_num_variant'
            , 'avg_num_variant'
        ],

        'between': [
            'betweenness'
        ],

        'reach': [
            'reachability_hap0'
            , 'reachability_hap1'
            , 'shortest_pos_path_hap0'
            , 'shortest_pos_path_hap1'
        ],

        'quality': [
            'min_qscore'
            , 'max_qscore'
            , 'avg_qscore'
        ],

        "bitmap": [
            'variant_bitmap'
        ]
    }
