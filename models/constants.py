from enum import Enum

class LossTypes(str, Enum):
    actor_loss = 'actor_loss'
    critic_loss = 'critic_loss'
    total_loss = 'total_loss'
    
def define_features(feature_settings):
    features = []
    ### MANDATORY FEATURES ###
    if 'dual' in feature_settings:
        features.append('cut_member_hap0')
        features.append('cut_member_hap1')
    elif 'reversible' in feature_settings:
        features.append('flip')

    ### optional features ###
    if 'variants' in feature_settings:
        features.append('n_variants')

    if 'neigh' in feature_settings:
        features.append('pos_neighbors')
        features.append('neg_neighbors')
        features.append('max_weight_node')
        features.append('min_weight_node')

    if 'arti' in feature_settings:
        features.append('is_articulation')
        
    if 'stats' in feature_settings:
        features.append('n_nodes')
        features.append('num_articulation')
        features.append('diameter')
        features.append('density')
        features.append('max_degree')
        features.append('min_degree')
        features.append('n_edges')
        features.append('node_connectivity')
        features.append('edge_connectivity')
        features.append('min_weight')
        features.append('max_weight')
        features.append('num_fragments')
        features.append('compression_factor')
        features.append('max_num_variant')
        features.append('min_num_variant')
        features.append('avg_num_variant')

    if 'between' in feature_settings:
        features.append('betweenness')

    if 'reach' in feature_settings:
        features.append('reachability_hap0')
        features.append('reachability_hap1')
        features.append('shortest_pos_path_hap0')
        features.append('shortest_pos_path_hap1')

    if 'quality' in feature_settings:
        features.append('min_qscore')
        features.append('max_qscore')
        features.append('avg_qscore')

    if "bitmap" in feature_settings:
        features.append('variant_bitmap')

    return features