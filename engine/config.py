import yaml
from enum import Enum
import torch
from pathlib import Path
import logging
import sys
import os
import wandb
import re

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

CONFIG_TYPE = Enum("CONFIG_TYPE", 'TRAIN TEST DATA_DESIGN FRAGS')

MAIN_LOG = "MAIN"
STATS_LOG_TRAIN = "STATS_TRAIN"
STATS_LOG_VALIDATE = "STATS_VALIDATE"
STATS_LOG_COLS_TRAIN = ['Episode',
                        'Reward',
                        'CutValue'
                        'Losses',
                        'Runtime']
STATS_LOG_COLS_VALIDATE = ['Descriptor', 'Episode', 'SumOfCuts', 'SumOfRewards', 'Switch Count', 'Mismatch Count', 'Flat Count', 'Phased Count', 'AN50', 'N50']

class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.experiment_dir = str(Path(config_file).parent.resolve())
        self.log_dir = self.experiment_dir + "/logs/"
        self.out_dir = self.experiment_dir + "/output/"

        # setup the experiment directory structure
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        
        # ...shared training and testing configs...
    def set_defaults(self):
        default_values = {
            'render': False,  # Enables the rendering of the environment
            'num_cores_torch': 4,  # Number of threads to use for Pytorch
            'device': "cpu",             
            'compress': True,
            'normalization': False,
            'debug': True,
            'epochs': 1,
            'node_features_dim': 3,
            'hidden_dim': [264, 264, 264],
            'light_logging': True,
            'id': "vanilla",
            'fragment_norm': False,
            'weight_norm': False,
            'clip': False,
            'features': ["dual", "between"]
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

        for f in os.listdir(self.out_dir):
            os.remove(os.path.join(self.out_dir, f))

        # set up performance tracking
        if self.log_wandb:
            wandb.init(project=self.project_name, entity="dphase", dir=self.log_dir, config=self, name=self.run_name)
        else:
            # automatically results in ignoring all wandb calls
            wandb.init(project=self.project_name, entity="dphase", dir=self.log_dir, id=config_file.id, mode="disabled")


        # logging
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                            handlers=[logging.FileHandler(self.log_dir + '/training.log', mode='w'),
                                      logging.StreamHandler(sys.stdout)])
    
        # enforce light logging if using multithreading validation
        if self.num_cores_validation > 1:
            self.light_logging = True

        if self.device == "cuda" or self.device == "cuda:0" or self.device == "cuda:1":
            self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def set_defaults(self):
        super().set_defaults()
        default_values = {
            'project_name': "dphase_experiments",
            'run_name': "vanilla",
            'panel_validation_frags': None, # Fragment files for validation
            'panel_validation_vcfs': None, # VCF files for validation
            'num_cores_validation': 8,
            'min_graph_size': 1,  # Minimum size of graphs to use for training
            'max_graph_size': 1000,  # Maximum size of graphs to use for training
            'skip_trivial_graphs': True,
            'skip_singleton_graphs': True,
            'seed': 12345,  # Random seed
            'max_episodes': None,  # Maximum number of episodes to run
            'render_view': "weighted_view",  # Controls how the graph is rendered
            'interval_validate': 500,  # Number of episodes between model validation runs
            'log_wandb': False,
            'ultra_light_mode': False,            
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
        
        if self.device == "cuda" or self.device == "cuda:0" or self.device == "cuda:1":
            self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu") 

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

    def set_defaults(self):
        default_values = {
            'shuffle': True,
            'seed': 1234,  # Random seed
            'num_samples_per_category_default': 1000,
            'drop_redundant': False,
            'global_ranges': {},
            'ordering_ranges': {},
            'save_indexes_path': None
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)


class FragmentConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        super().__init__(config_file)
        self.log_file_main = self.log_dir + 'fragments_main.log'
        file_handler = logging.FileHandler(self.log_file_main, mode='w')
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.getLevelName("INFO"),
                            handlers=[logging.StreamHandler(sys.stdout), file_handler])
        self.set_defaults()

    @staticmethod
    def get_defaults_short():
        return {
            'log_reads': False,
            'mapq': 20,
            'mbq': 13,
            'max_isize': 1000,
            'allow_supplementary': False,
            'allow_overlap': False,
            'max_coverage': None,
            'enable_read_selection': False,
            "max_snp_coverage": 1000000000,
            "min_coverage_to_filter_ref": 10000000,
            "min_coverage_to_filter_alt": 10000000,
            "realign_overhang": None,
            "min_highmapq_ratio": 0.0,
            "min_mapq1_ratio": 0.0,
            "max_bad_allele_ratio": 1.0,
            "min_alt_allele_ratio": 0.0,
            'enable_strand_filter': False,
            'reference': None,
        }

    @staticmethod
    def get_defaults_ont():
        return {
            'log_reads': False,
            'mapq': 20,
            'mbq': 0,
            'max_isize': None,
            'allow_supplementary': False,
            'allow_overlap': False,
            'max_coverage': None,
            'enable_read_selection': False,
            "max_bad_allele_ratio": 0.5,
            "min_alt_allele_ratio": 0.1,
            "max_snp_coverage": 150,
            "min_coverage_to_filter_ref": 10,
            "min_coverage_to_filter_alt": 10,
            "realign_overhang": 20,
            "min_highmapq_ratio": 0.1,
            "min_mapq1_ratio": 0.5,
            'enable_strand_filter': True,
        }

    @staticmethod
    def get_defaults_hifi():
        return {
            'log_reads': False,
            'mapq': 20,
            'mbq': 13,
            'max_isize': None,
            'allow_supplementary': False,
            'allow_overlap': False,
            'max_coverage': None,
            'enable_read_selection': False,
            "max_bad_allele_ratio": 0.5,
            "min_alt_allele_ratio": 0.1,
            "max_snp_coverage": 150,
            "min_coverage_to_filter_ref": 10,
            "min_coverage_to_filter_alt": 10,
            "realign_overhang": 50,
            "min_highmapq_ratio": 0.1,
            "min_mapq1_ratio": 0.5,
            'enable_strand_filter': False,
        }

    def set_defaults(self):
        if self.platform == "ONT":
            default_values = self.get_defaults_ont()
        elif self.platform == "hifi":
            default_values = self.get_defaults_hifi()
        elif self.platform == "illumina":
            default_values = self.get_defaults_short()
        else:
            print("Unexpected platform: " + self.platform)
            sys.exit(-1)
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)


def load_config(fname, config_type=CONFIG_TYPE.TRAIN):
    # Load a YAML configuration file
    with open(fname) as file:
        loader = yaml.FullLoader
        # expression enabling yaml to read floats in scientific notation
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        config = yaml.load(file, Loader=loader)
    if config_type == CONFIG_TYPE.TRAIN:
        return TrainingConfig(fname, **config)
    elif config_type == CONFIG_TYPE.TEST:
        return TestConfig(fname, **config)
    elif config_type == CONFIG_TYPE.DATA_DESIGN:
        return DataConfig(fname, **config)
    elif config_type == CONFIG_TYPE.FRAGS:
        return FragmentConfig(fname, **config)



