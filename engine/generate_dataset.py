import argparse
import logging

import graphs.graph_dataset
import random
import engine.config as config_utils

if __name__ == '__main__':
    # ------ CLI ------
    parser = argparse.ArgumentParser(description='Generate graph datasets for training and validation.')
    parser.add_argument('--config', help='Dataset configuration YAML')
    args = parser.parse_args()
    # -----------------

    config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.DATA_GENERATION)
    random.seed(config.seed)
    graphs.graph_dataset.GraphDataset(config)
