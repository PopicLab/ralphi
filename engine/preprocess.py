import argparse
import graphs.frag_graph
import random
import engine.config as config_utils

if __name__ == '__main__':
    # ------ CLI ------
    parser = argparse.ArgumentParser(description='Train haplotype phasing')
    parser.add_argument('--config', help='Training configuration YAML')
    parser.add_argument('--training_ordering_config', help='Training configuration YAML')
    parser.add_argument('--validation_ordering_config', help='Training configuration YAML')
    args = parser.parse_args()
    # -----------------

    # Load the config
    config = config_utils.load_config(args.config)
    training_config = config_utils.load_config(args.training_ordering_config, config_type=config_utils.CONFIG_TYPE.DATA_DESIGN)
    validation_config = config_utils.load_config(args.validation_ordering_config, config_type=config_utils.CONFIG_TYPE.DATA_DESIGN)

    random.seed(config.seed)
    graph_dataset = graphs.frag_graph.GraphDataset(config)
    global_df = graph_dataset.load_indices()
    if validation_config is not None:
        validation_df = graphs.frag_graph.GraphDataset(config, validation_config).load_indices()
        # Remove the validation graphs from the pool of graphs available for training
        global_df = global_df[~global_df.component_path.isin(validation_df.component_path)]

    if training_config is not None:
        training_dataset = graph_dataset.dataset_nested_design(global_df, training_config)