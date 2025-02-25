from enum import Enum


class LossTypes(str, Enum):
    actor_loss = 'actor_loss'
    critic_loss = 'critic_loss'
    total_loss = 'total_loss'


FEATURES_DICT = {
    'dual': [
        'cut_member_hap0',
        'cut_member_hap1'
    ],
    'between': [
        'betweenness'
    ],

    'min_weight': [
        'min_weight'
    ],
    'max_weight': [
        'max_weight'
    ],
    'n_edges': [
        'n_edges'
    ],
    'n_nodes': [
        'n_nodes'
    ],
    'density': [
        'density'
    ],
    'diameter': [
        'diameter'
    ],
    'n_variants': [
        'n_variants'
    ],
    'compression_factor': [
        'compression_factor'
    ],
    'n_articulation_points': [
        'articulation_points'
    ]
}
