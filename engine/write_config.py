import os

def update_dict(default_values, id_file, layer_type, layer_struct, attention_layer, other_vars):
    param = default_values.copy()
    param['id'] = id_file
    param['hidden_dim'] = layer_struct
    param['layer_type'] = layer_type
    param['embedding_vars']['attention_layer'] = attention_layer
    param['embedding_vars'] = {**param['embedding_vars'], ** other_vars}
    return param

def write_yaml(config_path, id_file, parameters):
    if not os.path.exists(config_path + id_file):
        os.mkdir(config_path + id_file)
    with open(config_path + id_file + '/config.yaml', "w") as file:
        file.write('\n'.join("{}: {}".format(k, v) for k, v in parameters.items()))


def generate_files(config_path, panel_path, frags_path, vcfs_path):
    layers_number = [3, 4, 5, 10]
    default_layers = [132, 264, [132, 264]]
    layer_types = ["gcn", "gin", "pna"]
    attention = [None, 0]
    gat_residual = [None, True]
    gat_num_heads = [1, 2, 3, 4, 5]
    gcn_bias = [None, True]
    gin_aggregator = ['sum', 'mean', 'max']
    pna_aggregator = [["sum", "mean", "max", "min", "std"], ["sum", "mean", "std"]]
    pna_scaler = [["identity", "amplification", "attenuation"]]
    pna_residual = [None, True]

    default_values = {
        'panel': panel_path,
        'panel_validation_frags': frags_path,
        'panel_validation_vcfs': vcfs_path,
        'min_graph_size': 1,  # Minimum size of graphs to use for training
        'max_graph_size': 1000,  # Maximum size of graphs to use for training
        'skip_trivial_graphs': True,
        'skip_singleton_graphs': True,
        'seed': 12345,  # Random seed
        'max_episodes': 'null',  # Maximum number of episodes to run
        'render': False,  # Enables the rendering of the environment
        'render_view': "weighted_view",  # Controls how the graph is rendered
        'num_cores': 4,  # Number of threads to use for Pytorch
        'interval_validate': 500,  # Number of episodes between model validation runs
        'debug': True,
        'log_wandb': True,
        'id': None,
        'compress': True,
        'normalization': False,
        # caching parameters
        'load_components': True,
        'store_components': True,
        'store_indexes': True,
        # model parameters
        'pretrained_model': 'null',  # path to pretrained model; null or "path/to/model"
        'in_dim': 1,
        'hidden_dim': None,
        'layer_type': None,
        'embedding_vars': {},
        'gamma': 0.98,
        'lr': 0.00003
    }
    for layer_type in layer_types:
        for num_layer in layers_number:
            for version, layer_dim in enumerate(default_layers):
                if isinstance(layer_dim, list):
                    layer_struct = [layer for layer in layer_dim]
                    layer_struct += [layer_dim[-1]] * (num_layer - len(layer_dim))
                else:
                    layer_struct = [layer_dim] * num_layer
                for attention_layer in attention:
                    id_attn = ''
                    if not attention_layer is None:
                        id_attn = 'no_attn'
                        attention_layer = [[0]] * num_layer
                    if layer_type == 'gcn':
                        id_bias = ''
                        for bias in gcn_bias:
                            if not bias is None:
                                id_bias = 'bias'
                                bias = [bias] * num_layer
                            id_file = "_". join([layer_type, str(num_layer), str(version), id_attn, id_bias])
                            parameters = update_dict(default_values, id_file
                                                     , layer_type, layer_struct, attention_layer,
                                                     {'bias': bias})
                            write_yaml(config_path, id_file, parameters)
                    elif layer_type == 'gin':
                        for aggreg in gin_aggregator:
                            id_file = "_".join([layer_type, str(num_layer), str(version), aggreg])
                            aggreg = [aggreg] * num_layer
                            parameters = update_dict(default_values, id_file,
                                                     layer_type, layer_struct, attention_layer,
                                                     {'aggregator_type': aggreg})
                            write_yaml(config_path, id_file, parameters)
                    elif layer_type == 'pna':
                        for version_aggreg, aggreg in enumerate(pna_aggregator):
                            aggreg = [aggreg] * num_layer
                            for version_scaler,scaler in enumerate(pna_scaler):
                                scaler = [scaler] * num_layer
                                for residual in pna_residual:
                                    id_res = ''
                                    if not residual is None:
                                        residual = [residual] * num_layer
                                        id_res = 'res'
                                    id_file = "_". join([layer_type, str(num_layer), str(version), str(version_aggreg), str(version_scaler), id_res])
                                    parameters = update_dict(default_values, id_file,
                                                             layer_type, layer_struct, attention_layer,
                                                             {'aggregator': aggreg, 'scaler': scaler,
                                                              'residual': residual})
                                    write_yaml(config_path, id_file, parameters)





if __name__ == "__main__":
    import sys

    generate_files(*sys.argv[1:])
