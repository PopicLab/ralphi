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

CONFIG_TYPE = Enum("CONFIG_TYPE", 'TRAIN TEST DATA_GENERATION DATA_SELECTION')

MAIN_LOG = "MAIN"
STATS_LOG_TRAIN = "STATS_TRAIN"
STATS_LOG_VALIDATE = "STATS_VALIDATE"
STATS_LOG_COLS_TRAIN = ['Episode', 'Reward', 'CutValue' 'Losses', 'Runtime']
STATS_LOG_COLS_VALIDATE = ['Descriptor', 'Episode', 'SumOfCuts', 'SumOfRewards', 'Switch Count', 'Mismatch Count',
                           'Flat Count', 'Phased Count', 'AN50', 'N50']


SHARED_DEFAULTS = {
    'logging_level': "INFO",
    'n_procs': 1,
    'num_cores_torch': 4,
    'device': "cpu",
    'debug': True,
    'seed': 1234,
    'team_name': 'ralphi',
    'project_name': "ralphi",
    'run_name': "ralphi",
    'log_wandb': False,
}

MODEL_DEFAULTS = {
    'node_features_dim': 3,
    'hidden_dim': [264],
    'layer_type': "gcn",
    'embedding_vars': {'attention_layer': [[0]]},
    'features': ["dual", "between"],
    'fragment_norm': False,
    'weight_norm': False,
    'clip': False,
    'clip_node': False,
    'advantage': "nstep",
    'normalization': False,
    'flip': False
}

SHARED_DATA_DEFAULTS = {
    'mapq': 20,
    'no_filter': False,
    'chr_names': None,
    'compress': True,
    'approximate_betweenness': True,
    'num_pivots': 10,
    "min_highmapq_ratio": 0,
    'supp_distance_th': 1000000,
    'log_reads': False,
}

DATA_DEFAULTS_SHORT = {
    'reference': None,
    'mbq': 13,
    "realign_overhang": None,
    'filter_bad_reads': False,
    'enable_read_selection': False,
    "max_snp_coverage": float('inf'),
    'require_hiqh_mapq': False,
    "min_coverage_to_filter": float('inf'),
    'enable_strand_filter': False,
    'max_isize': 1000,
    'max_discordance': 1.0,
    'skip_post': False,
    'read_overlap_th': None,
}


DATA_DEFAULTS_LONG = {
    "mbq": 0,
    "realign_overhang": 10,
    'filter_bad_reads': True,
    'max_discordance': 0.1,
    'enable_read_selection': True,
    'max_coverage': 15,
    "max_snp_coverage": 200,
    'require_hiqh_mapq': True,
    "min_coverage_to_filter": 8,
    "min_coverage_strand": 10,
    'skip_post': True,
    'enable_strand_filter': True,
    'read_overlap_th': 100,
}

PHASE_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **SHARED_DATA_DEFAULTS}
PHASE_DEFAULTS.update({
    'max_graph_size': float('inf'),
    'skip_singleton_graphs': False,
    'skip_trivial_graphs': False,
    'test_mode': True,
    'load_components': False,
    'store_components': False,
    'store_indexes': False,
})

DATA_GENERATION_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **SHARED_DATA_DEFAULTS}
DATA_GENERATION_DEFAULTS.update({
    'test_mode': False,
    'drop_chr': ['chr20'],
    'skip_singleton_graphs': True,
    'skip_trivial_graphs': True,
    'size': None,
    'validation_ratio': 0.1,
    'selection_config': None,
})

TRAIN_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **SHARED_DATA_DEFAULTS}
TRAIN_DEFAULTS.update({
    'test_mode': False,
    'gamma': 0.98,
    'lr': 0.000003,
    'epochs': 1,
    'n_procs': 8,
    'max_episodes': None, # maximum number of episodes to run
    'interval_episodes_to_validation': 30000,  # number of episodes between model validation runs
    'render_view': "weighted_view",
    'load_components': True,
    'store_components': True,
    'store_indexes': True,
    'pretrained_model': None,
})


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        self.experiment_dir = str(Path(config_file).parent.resolve())
        self.log_dir = self.experiment_dir + "/logs/"
        self.out_dir = self.experiment_dir + "/output/"

        # setup the experiment directory structure
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        if self.chr_names is None:
            self.chr_names = ['chr{}'.format(x) for x in range(1, 23)]

        # logging
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.getLevelName(self.logging_level),
                            handlers=[logging.FileHandler(self.log_dir + '/main.log', mode='w'),
                                      logging.StreamHandler(sys.stdout)])

    def set_defaults(self, default_values_dict):
        for k, v, in default_values_dict.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)

    def __str__(self):
        s = " ===== Config =====\n"
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s

class PhaseConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        if self.platform == "illumina":
            self.set_defaults(DATA_DEFAULTS_SHORT)
        else:
            self.set_defaults(DATA_DEFAULTS_LONG)
        self.set_defaults(PHASE_DEFAULTS)
        self.__dict__.update(entries)
        super().__init__(config_file)

        if self.chr_names is None:
            self.chr_names = ['chr{}'.format(x) for x in range(1, 23)]
        self.device = torch.device("cpu")
        self.output_vcf = self.out_dir + "/ralphi.vcf"
        logging.info(self)


class TrainingConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.set_defaults(TRAIN_DEFAULTS)
        super().__init__(config_file)
        self.model_path = self.out_dir + "/ralphi_model_final.pt"
        self.best_model_path = self.out_dir + "/ralphi_model_best.pt"
        self.validation_output_vcf = self.out_dir + "/validation_output_vcf.vcf"

        for f in os.listdir(self.out_dir):
            os.remove(os.path.join(self.out_dir, f))

        # set up performance tracking
        if self.log_wandb:
            wandb.init(project=self.project_name, entity=self.team_name, dir=self.log_dir, config=self, name=self.run_name)
        else:
            # automatically results in ignoring all wandb calls
            wandb.init(project=self.project_name, entity=self.team_name, dir=self.log_dir, mode="disabled")

        # logging
        logging.info(self)

        # enforce light logging if using multithreading validation
        if self.n_procs > 1:
            self.light_logging = True

        if self.device == "cuda" or self.device == "cuda:0" or self.device == "cuda:1":
            self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")


class DataConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        if self.platform == "illumina":
            self.set_defaults(DATA_DEFAULTS_SHORT)
        else:
            self.set_defaults(DATA_DEFAULTS_LONG)
        self.set_defaults(DATA_GENERATION_DEFAULTS)
        super().__init__(config_file)

        for f in os.listdir(self.out_dir):
            os.remove(os.path.join(self.out_dir, f))

        # logging
        logging.info(self)

        if self.selection_config:
            self.selection_config = load_config(self.selection_config,
                                                config_type=CONFIG_TYPE.DATA_SELECTION)
            # Copy the selection config used in the local folder
            with open(self.experiment_dir + '/selection_config.yaml', 'w') as outfile:
                yaml.safe_dump(self.selection_config.__dict__, outfile, default_flow_style=False)


class SelectionConfig:
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.set_defaults()

    def set_defaults(self):
        default_values = {
            'shuffle': False,
            'Global': {},
        }
        for k, v, in default_values.items():
            if not hasattr(self, k):
                self.__setattr__(k, v)


def load_config(fname, config_type=CONFIG_TYPE.TRAIN):
    # Load a YAML configuration file
    if fname is None: return None
    with open(fname) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config_type == CONFIG_TYPE.TRAIN:
        return TrainingConfig(fname, **config)
    elif config_type == CONFIG_TYPE.TEST:
        return PhaseConfig(fname, **config)
    elif config_type == CONFIG_TYPE.DATA_GENERATION:
        return DataConfig(fname, **config)
    elif config_type == CONFIG_TYPE.DATA_SELECTION:
        return SelectionConfig(fname, **config)
