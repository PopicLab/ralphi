import yaml
from enum import Enum
import torch
from pathlib import Path
import logging
import sys
import os
import wandb

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

CONFIG_TYPE = Enum("CONFIG_TYPE", 'TRAIN TEST DATA_DESIGN')

MAIN_LOG = "MAIN"
STATS_LOG_TRAIN = "STATS_TRAIN"
STATS_LOG_VALIDATE = "STATS_VALIDATE"
STATS_LOG_COLS_TRAIN = ['Episode',
                        'Reward',
                        'CutValue'
                        'Losses',
                        'GraphProperties',
                        'Runtime']
STATS_LOG_COLS_VALIDATE = ['Descriptor', 'Episode', 'SumOfCuts', 'SumOfRewards', 'Switch Count', 'Mismatch Count', 'Flat Count', 'Phased Count', 'AN50', 'N50']

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.experiment_dir = str(Path(config_file).parent.resolve())
        self.device = torch.device("cpu")
        self.log_dir = self.experiment_dir + "/logs/"
        self.out_dir = self.experiment_dir + "/output/"

        # setup the experiment directory structure
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        # ...shared training and testing configs...
    def set_defaults(self):
        default_values = {
            'render': False,  # Enables the rendering of the environment
            'num_cores': 4,  # Number of threads to use for Pytorch
            'compress': True,
            'normalization': False,
            'debug': True,
            'in_dim': 1,
            'hidden_dim': 264,
            'num_layers': 3,
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)

    def __str__(self):
        s = " ===== Config =====\n"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class TrainingConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.set_defaults()
        super().__init__(config_file)
        self.model_path = self.out_dir + "/dphase_model_final.pt"
        self.best_model_path = self.out_dir + "/dphase_model_best.pt"
        self.validation_output_vcf = self.out_dir + "/validation_output_vcf.vcf"

        # logging
        # main log file
        self.log_file_main = self.log_dir + 'main.log'
        file_handler = logging.FileHandler(self.log_file_main, mode='w')
        file_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(file_handler.formatter)
        logger_main = logging.getLogger(MAIN_LOG)
        logger_main.setLevel(level=logging.INFO)
        logger_main.addHandler(file_handler)
        logger_main.addHandler(stream_handler)
        # training stats log file
        self.log_file_stats_train = self.log_dir + 'train_episodes_stats.csv'
        file_handler = logging.FileHandler(self.log_file_stats_train, mode='w')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger_stats_train = logging.getLogger(STATS_LOG_TRAIN)
        logger_stats_train.setLevel(level=logging.INFO)
        logger_stats_train.addHandler(file_handler)
        logger_stats_train.info(",".join(STATS_LOG_COLS_TRAIN))
        # validation stats log file
        self.log_file_stats_validate = self.log_dir + 'validate_episodes_stats.csv'
        file_handler = logging.FileHandler(self.log_file_stats_validate, mode='w')
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger_stats_validate = logging.getLogger(STATS_LOG_VALIDATE)
        logger_stats_validate.setLevel(level=logging.INFO)
        logger_stats_validate.addHandler(file_handler)
        logger_stats_validate.info(",".join(STATS_LOG_COLS_VALIDATE))
        logger_main.info(self)

        if os.path.exists(self.out_dir + "/benchmark.txt"):
            os.remove(self.out_dir + "/benchmark.txt")

        # set up performance tracking
        if self.log_wandb:
            wandb.init(project="data_orderings", entity="dphase", dir=self.log_dir)
        else:
            # automatically results in ignoring all wandb calls
            wandb.init(project="data_ordering", entity="dphase", dir=self.log_dir, mode="disabled")

        # logging
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                            handlers=[logging.FileHandler(self.log_dir + '/training.log', mode='w'),
                                      logging.StreamHandler(sys.stdout)])
    def __str__(self):
        s = " ===== Config =====\n"
        s += '\n'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s

    def set_defaults(self):
        super().set_defaults()
        default_values = {
            'panel_validation_frags': None, # Fragment files for validation
            'panel_validation_vcfs': None, # VCF files for validation
            'min_graph_size': 1,  # Minimum size of graphs to use for training
            'max_graph_size': 1000,  # Maximum size of graphs to use for training
            'skip_trivial_graphs': True,
            'skip_singleton_graphs': True,
            'seed': 12345,  # Random seed
            'max_episodes': None,  # Maximum number of episodes to run
            'render_view': "weighted_view",  # Controls how the graph is rendered
            'interval_validate': 500,  # Number of episodes between model validation runs
            'log_wandb': False,
            # caching parameters
            'load_components': True,
            'store_components': True,
            'store_indexes': True,
            # model parameters
            'pretrained_model': None,  # path to pretrained model; null or "path/to/model"
            'gamma': 0.98,
            'lr': 0.00003
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)


class TestConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.set_defaults()
        super().__init__(config_file)
        self.phasing_output_path = self.out_dir + "phasing_output.pickle"
        self.output_vcf = self.out_dir + "dphase_phased.vcf"

        # logging
        # noinspection PyArgumentList
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                            handlers=[logging.FileHandler(self.log_dir + '/main.log', mode='w'),
                                      logging.StreamHandler(sys.stdout)])
        logging.info(self)

    def __str__(self):
        s = " ===== Config =====\n"
        s += '\n'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s
    def set_defaults(self):
        super().set_defaults()
        default_values = {
            'min_graph_size': 1,
            'max_graph_size': float('inf'),
            'skip_singleton_graphs': True,
            'skip_trivial_graphs': False,
            # caching parameters
            'load_components': False,
            'store_components': False,
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)

class DataConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.set_defaults()
        super().__init__(config_file)

    def __str__(self):
        s = " ===== Config =====\n"
        s += '\n'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s
    def set_defaults(self):
        #super().set_defaults()
        default_values = {
            'min_n_nodes': 1,
            'max_n_nodes': float('inf'),
            'min_n_edges': 1,
            'max_n_edges': float('inf'),
            'min_density': 0,
            'max_density': 1,
            'min_articulation_points': 0,
            'max_articulation_points': float('inf'),
            'min_diameter': 1,
            'max_diameter': float('inf'),
            'min_node_connectivity': 1,
            'max_node_connectivity': float('inf'),
            'min_edge_connectivity': 1,
            'max_edge_connectivity': float('inf'),
            'min_min_degree': 1,
	    'max_min_degree': float('inf'),
            'min_max_degree': 1,
            'max_max_degree': float('inf'),
	    'shuffle': True,
            'seed': 1234,  # Random seed
            'num_samples': 100000,
            'num_samples_per_category': 200,
            'epochs': 1,
            'save_indexes': False
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)


def load_config(fname, config_type=CONFIG_TYPE.TRAIN):
    # Load a YAML configuration file
    with open(fname) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config_type == CONFIG_TYPE.TRAIN:
        return TrainingConfig(fname, **config)
    elif config_type == CONFIG_TYPE.TEST:
        return TestConfig(fname, **config)
    elif config_type == CONFIG_TYPE.DATA_DESIGN:
        return DataConfig(fname, **config)



