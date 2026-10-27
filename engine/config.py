import yaml
from enum import Enum
from types import SimpleNamespace
import torch
from pathlib import Path
import logging
import sys
import os
import wandb

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

CONFIG_TYPE = Enum("CONFIG_TYPE", 'TRAIN TEST DATA_GENERATION DATA_SELECTION LINK')

MAIN_LOG = "MAIN"
STATS_LOG_TRAIN = "STATS_TRAIN"
STATS_LOG_VALIDATE = "STATS_VALIDATE"
STATS_LOG_COLS_TRAIN = ['Episode', 'Reward', 'CutValue' 'Losses', 'Runtime']
STATS_LOG_COLS_VALIDATE = ['Descriptor', 'Episode', 'SumOfCuts', 'SumOfRewards', 'Switch Count', 'Mismatch Count',
                           'Flat Count', 'Phased Count', 'AN50', 'N50']

REQUIRED_GLOBAL_PARAMS = {'vcf', 'inputs'}
REQUIRED_DATA_PARAMS = {'bam', 'platform'}

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
    'skip_post': True,
    'reference': None,
}

MODEL_DEFAULTS = {
    'model': None,
    'node_features_dim': 3,
    'hidden_dim': [264],
    'layer_type': "gcn",
    'embedding_vars': {'attention_layer': [[0]]},
    'features': ["dual", "between"],
    'fragment_norm': False,
    'weight_norm': False,
    'clip': False,
    'normalization': False,
    'flip': False,
}

GRAPH_DEFAULTS = {
    'chr_names': None,
    'compress': True,
    'approximate_betweenness': True,
    'num_pivots': 10,
}

SHARED_DATA_DEFAULTS = {
    'mapq': 20,
    'no_filter': False,
    "min_highmapq_ratio": 0,
    'supp_distance_th': 1000000,
    'log_reads': False,
    'is_hic': False,
    'min_isize': 0,
    'max_isize': float('inf'),
    'realign': False
}

DATA_DEFAULTS_SHORT = {
    'mbq': 13,
    "realign_overhang": None,
    'filter_bad_reads': False,
    'enable_read_selection': False,
    "max_snp_coverage": float('inf'),
    'require_high_mapq': False,
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
    'require_high_mapq': True,
    "min_coverage_to_filter": 8,
    "min_coverage_strand": 10,
    'skip_post': True,
    'enable_strand_filter': True,
    'read_overlap_th': 100,
    'realign': True,
}

DATA_DEFAULTS_HIC= {
    "mbq": 13,
    "realign_overhang": None,
    'filter_bad_reads': False,
    'max_discordance': 1.0,
    'enable_read_selection': False,
    "max_snp_coverage": 200,
    'require_high_mapq': False,
    "min_coverage_to_filter": float('inf'),
    'enable_strand_filter': False,
    'read_overlap_th': None,
    'is_hic': True,
    'skip_post': True,
    'min_isize': 1000,
}


PHASE_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **GRAPH_DEFAULTS}
PHASE_DEFAULTS.update({
    'max_graph_size': float('inf'),
    'skip_singleton_graphs': False,
    'skip_trivial_graphs': False,
    'test_mode': True,
    'load_components': False,
})

DATA_GENERATION_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **GRAPH_DEFAULTS}
DATA_GENERATION_DEFAULTS.update({
    'test_mode': False,
    'drop_chr': ['chr1','chr20'],
    'skip_singleton_graphs': True,
    'skip_trivial_graphs': True,
    'size': None,
    'validation_ratio': 0.1,
    'selection_config': None,
})

TRAIN_DEFAULTS = {**SHARED_DEFAULTS, **MODEL_DEFAULTS, **GRAPH_DEFAULTS}
TRAIN_DEFAULTS.update({
    'test_mode': False,
    'gamma': 0.98,
    'lr': 0.00003,
    'epochs': 1,
    'n_procs': 8,
    'max_episodes': None, # maximum number of episodes to run
    'interval_episodes_to_validation': 500,  # number of episodes between model validation runs
    'load_components': True,
    'pretrained_model': None,
})

DATA_PARAMS = {
    *SHARED_DATA_DEFAULTS, 
    *DATA_DEFAULTS_SHORT, 
    *DATA_DEFAULTS_LONG,
    *DATA_DEFAULTS_HIC,
    *REQUIRED_DATA_PARAMS
}

GLOBAL_PARAMS = {
    *TRAIN_DEFAULTS,
    *DATA_GENERATION_DEFAULTS, 
    *PHASE_DEFAULTS,
    *MODEL_DEFAULTS,
    *REQUIRED_GLOBAL_PARAMS
}


class LogNamespace(SimpleNamespace):
    def __repr__(self):
        return repr(self.__dict__)
    

PLATFORM_PRESETS = {
    "illumina": DATA_DEFAULTS_SHORT,
    "ont": DATA_DEFAULTS_LONG,
    "hifi": DATA_DEFAULTS_LONG,
    "hic": DATA_DEFAULTS_HIC
}


def platform_preset(platform):
    return PLATFORM_PRESETS[platform]


def resolve_input(entry, global_entries):
    if any(required_param not in entry for required_param in REQUIRED_DATA_PARAMS):
        raise ValueError(f'Each input section must contain the parameters {REQUIRED_DATA_PARAMS}, please check the input {entry}')    
    # Get the platform defaults
    resolved = dict(SHARED_DATA_DEFAULTS)
    resolved.update(platform_preset(entry['platform']))
    # Ensure the parameters in entry are valid for the inputs section
    provided_keys =  set(entry.keys())
    if not provided_keys.issubset(DATA_PARAMS):
        invalid_keys = provided_keys - DATA_PARAMS
        raise ValueError(f"Parameters invalid for the `inputs` section: {invalid_keys}")
    # Update with the input parameters
    resolved.update(entry)
    if entry['platform'] in ["ont", 'hifi'] and not resolved['realign']:
        logging.warning("realign is set to False for platform %s; " \
        "realignment is recommended for ont and hifi platforms; " \
        "realignment will be skipped.", entry)
    if resolved['enable_read_selection'] and resolved['read_overlap_th'] is None:
        if entry['platform'] == 'illumina':
            raise ValueError('Please note that downsampling is not recommended for illumina reads, if you use '\
                             f'enable_read_selection please specify read_overlap_th as the default is None. Issue in {entry}')
        raise ValueError(f'read_overlap_th must be specified if enable_read_selection, but got {entry}')
    if resolved['realign'] and (not hasattr(global_entries, 'reference') or not global_entries.reference):
        raise ValueError(f'A reference must be specified when realign is set to True, realign is used for {entry}')
    return LogNamespace(**resolved)


def resolve_inputs(config):
    if not config.inputs:
        raise ValueError('The list of inputs provided is empty')
    config.inputs = [resolve_input(entry, config) for entry in config.inputs]


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
        self.set_defaults(PHASE_DEFAULTS)
        self.__dict__.update(entries)
        super().__init__(config_file)
        resolve_inputs(self)
        # skip_post has a platform dependent default while not being a data processing parameter
        # We check if their is only one BAM file provided and if its platform is illumina to set the default 
        if 'skip_post' not in entries and  len(self.inputs) == 1 and self.inputs[0].platform == 'illumina':
            self.skip_post = False
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

        if self.device == "cuda" or self.device == "cuda:0" or self.device == "cuda:1":
            self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")


class DataConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        self.set_defaults(DATA_GENERATION_DEFAULTS)
        super().__init__(config_file)
        resolve_inputs(self)
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
    # Check for invalid input keys
    provided_keys = set(config.keys())
    if not provided_keys.issubset(GLOBAL_PARAMS):
        invalid_keys = provided_keys - GLOBAL_PARAMS
        raise ValueError(f"Invalid global parameters provided: {invalid_keys}")
    if any(required_param not in provided_keys for required_param in REQUIRED_GLOBAL_PARAMS):
        missing_keys = REQUIRED_GLOBAL_PARAMS - provided_keys
        raise ValueError(f'Missing global parameters {missing_keys}.')
    if config_type == CONFIG_TYPE.TRAIN:
        return TrainingConfig(fname, **config)
    elif config_type == CONFIG_TYPE.TEST:
        return PhaseConfig(fname, **config)
    elif config_type == CONFIG_TYPE.DATA_GENERATION:
        return DataConfig(fname, **config)
    elif config_type == CONFIG_TYPE.DATA_SELECTION:
        return SelectionConfig(fname, **config)
