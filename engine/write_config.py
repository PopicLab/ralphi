import os
import models.constants as constants


def update_dict(default_values, id_file, layer_type, layer_struct, attention_layer, other_vars):
    param = default_values.copy()
    param['embedding_vars'] = {}
    param['run_name'] = id_file
    param['hidden_dim'] = layer_struct
    param['layer_type'] = layer_type
    if attention_layer is not None:
        param['embedding_vars']['attention_layer'] = attention_layer
    other_vars = {key: value for key, value in other_vars.items() if value is not None}
    param['embedding_vars'] = {**param['embedding_vars'], **other_vars}
    return param


def write_yaml(config_path, id_file, parameters):
    if not os.path.exists(config_path + id_file):
        os.mkdir(config_path + id_file)
    with open(config_path + id_file + '/config.yaml', "w") as file:
        file.write('\n'.join("{}: {}".format(k, v) for k, v in parameters.items()))

def count_features(features):
    count = list(len(feature_list[feature_name]) for feature_list in constants.NodeFeatures
                 for feature_name in features)
    return str(sum(count))

def generate_files(config_path, panel_path, frags_path, vcfs_path=None):
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    project = "balanced_training_and_validation"
    layers_number = [1, 3, 2]
    # default_layers = [264, 528, [132, 264, 528], [33, 66, 132, 264, 528]]
    default_layers = [264, 528]
    # layer_types = ["gcn", "gin", "pna"]
    layer_types = ["gcn", "gin", "pna", "gat"]
    attention = [0]
    gat_residual = [None, True]
    gat_num_heads = [1, 3, 5]
    gcn_bias = [None]
    gin_aggregator = ['sum']
    pna_aggregator = [["sum", "mean", "std"]]
    pna_scaler = [["identity", "amplification", "attenuation"]]
    pna_residual = [None, True]
    features = ["dual", "reach"]
    num_features = count_features(features)
    num_cores_torch = 2
    num_cores_validation = 4
    weight_norm = False
    fragment_norm = False
    clip = True
    lr = 0.00003
    run_name_basis = "_".join([num_features, "_".join(features), "lr", str(lr)])
    if weight_norm:
        run_name_basis += "_norma"
    if fragment_norm:
        run_name_basis += "_frag"
    if clip:
        run_name_basis += "_clip"
    default_values = {
        'project_name': project,
        'panel': panel_path,
        'panel_validation_frags': frags_path,
        'panel_validation_vcfs': vcfs_path,
        'min_graph_size': 1,  # Minimum size of graphs to use for training
        'max_graph_size': 5000,  # Maximum size of graphs to use for training
        'skip_trivial_graphs': True,
        'skip_singleton_graphs': True,
        'seed': 12345,  # Random seed
        'max_episodes': 'null',  # Maximum number of episodes to run
        'render': False,  # Enables the rendering of the environment
        'render_view': "weighted_view",  # Controls how the graph is rendered
        'num_cores_torch': num_cores_torch,  # Number of threads to use for Pytorch
        'num_cores_validation': num_cores_validation,
        'interval_validate': 1000,  # Number of episodes between model validation runs
        'debug': True,
        'log_wandb': True,
        'run_name': None,
        'compress': True,
        'normalization': False,
        # caching parameters
        'load_components': True,
        'store_components': True,
        'store_indexes': True,
        # model parameters
        'pretrained_model': 'null',  # path to pretrained model; null or "path/to/model"
        'in_dim': num_features,
        'hidden_dim': None,
        'layer_type': None,
        'embedding_vars': {},
        'gamma': 0.98,
        'lr': lr,
        'weight_norm': weight_norm,
        'fragment_norm': fragment_norm,
        'clip': clip,
        'features': features,
        'light_logging': True
    }
    for layer_type in layer_types:
        for num_layer in layers_number:
            for layer_dim in default_layers:
                if isinstance(layer_dim, list):
                    if len(layer_dim) > num_layer:
                        continue
                    layer_id = str(layer_dim[0]) + "_" + str(len(layer_dim))
                    layer_struct = [layer for layer in layer_dim]
                    layer_struct += [layer_dim[-1]] * (num_layer - len(layer_dim))
                else:
                    layer_id = str(layer_dim)
                    layer_struct = [layer_dim] * num_layer
                for attention_layer in attention:
                    id_attn = ''
                    attention_struct = None
                    if attention_layer is not None:
                        id_attn = 'no_attn'
                        attention_struct = [[0]] * num_layer
                    if layer_type == 'gcn':
                        for bias in gcn_bias:
                            id_bias = ''
                            bias_list = None
                            if bias is not None:
                                id_bias = 'bias'
                                bias_list = [bias] * num_layer
                            id_file = "_".join([run_name_basis, layer_type, str(num_layer), layer_id, id_attn, id_bias])
                            parameters = update_dict(default_values, id_file
                                                     , layer_type, layer_struct, attention_struct,
                                                     {'bias': bias_list})
                            write_yaml(config_path, id_file, parameters)
                    elif layer_type == 'gin':
                        for aggreg in gin_aggregator:
                            id_file = "_".join([run_name_basis, layer_type, str(num_layer), layer_id, id_attn, aggreg])
                            aggreg = [aggreg] * num_layer
                            parameters = update_dict(default_values, id_file,
                                                     layer_type, layer_struct, attention_struct,
                                                     {'aggregator_type': aggreg})
                            write_yaml(config_path, id_file, parameters)
                    elif layer_type == 'pna':
                        for version_aggreg, aggreg in enumerate(pna_aggregator):
                            aggreg = [aggreg] * num_layer
                            for version_scaler, scaler in enumerate(pna_scaler):
                                scaler = [scaler] * num_layer
                                for residual in pna_residual:
                                    id_res = ''
                                    res_list = None
                                    if residual is not None:
                                        res_list = [residual] * num_layer
                                        id_res = 'res'
                                    id_file = "_".join([run_name_basis, layer_type, str(num_layer), layer_id, id_attn, str(version_aggreg),
                                                        str(version_scaler), id_res])
                                    parameters = update_dict(default_values, id_file,
                                                             layer_type, layer_struct, attention_struct,
                                                             {'aggregator': aggreg, 'scaler': scaler,
                                                              'residual': res_list})
                                    write_yaml(config_path, id_file, parameters)
                    elif layer_type == 'gat':
                        for residual in gat_residual:
                            id_res = ''
                            if residual is not None:
                                id_res = 'res'
                                residual = [True] * num_layer
                            for num_heads in gat_num_heads:
                                id_heads = 'heads_' + str(num_heads)
                                num_heads = [num_heads] * (num_layer - 1)
                                num_heads += [1]
                                if sum([layer_struct[i] % num_heads[i] != 0 for i in range(num_layer)]) > 0:
                                    continue
                                id_file = "_".join(
                                    [run_name_basis, layer_type, str(num_layer), layer_id, id_attn, id_res, id_heads])
                                parameters = update_dict(default_values, id_file,
                                                         layer_type, layer_struct, attention_struct,
                                                         {'num_heads': num_heads, 'residual': residual})
                                write_yaml(config_path, id_file, parameters)




if __name__ == "__main__":
    import sys

    generate_files(*sys.argv[1:])
