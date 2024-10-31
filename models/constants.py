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
    ]
}
