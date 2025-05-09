import argparse
import logging

import graphs.frag_graph
import random
import engine.config as config_utils

if __name__ == '__main__':
    # ------ CLI ------
    parser = argparse.ArgumentParser(description='Filter and order set of graphs to design training and validation datasets.')
    parser.add_argument('--config', help='Dataset configuration YAML')
    parser.add_argument('--training_config', help='Training configuration YAML')
    parser.add_argument('--validation_config', help='Validation configuration YAML')
    args = parser.parse_args()
    # -----------------

    # Load the config
    config = config_utils.load_config(args.config)
    training_config = config_utils.load_config(args.training_config, config_type=config_utils.CONFIG_TYPE.DATA_DESIGN)
    validation_config = config_utils.load_config(args.validation_config, config_type=config_utils.CONFIG_TYPE.DATA_DESIGN)

    random.seed(config.seed)
    graph_dataset = graphs.frag_graph.GraphDataset(config)
    global_df = graph_dataset.load_indices()
    logging.info('Overall the dataset contains %d graphs' % global_df.shape[0])
    if validation_config is not None:
        validation_df = graphs.frag_graph.GraphDataset(config, validation_config).load_indices()
        logging.info('Validation dataset number of graphs %d' % validation_df.shape[0])
        # Remove the validation graphs from the pool of graphs available for training
        global_df = global_df[~global_df.component_path.isin(validation_df.component_path)]
        logging.info('Remaining %d graphs available for training' % global_df.shape[0])

    if training_config is not None:
        training_dataset = graph_dataset.dataset_nested_design(global_df, training_config)
        logging.info('Training dataset number of graphs %d' % training_dataset.shape[0])