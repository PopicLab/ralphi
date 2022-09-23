import yaml
from enum import Enum
import torch
from pathlib import Path
import logging
import sys
import engine.constants as constants

logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.WARNING)

CONFIG_TYPE = Enum("CONFIG_TYPE", 'TRAIN TEST')

MAIN_LOG = "MAIN"
STATS_LOG_TRAIN = "STATS_TRAIN"
STATS_LOG_VALIDATE = "STATS_VALIDATE"
STATS_LOG_COLS_TRAIN = ['Episode',
                        'Reward',
                        'ActorLoss',
                        'CriticLoss',
                        'SumLoss',
                        constants.GraphStats.num_nodes,
                        constants.GraphStats.num_edges,
                        constants.GraphStats.cut_value,
                        'Runtime']
STATS_LOG_COLS_VALIDATE = ['Episode', 'SumOfCuts', 'SumOfRewards']

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
        logging.info(self)

    def __str__(self):
        s = " ===== Config ====="
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class TrainingConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        super().__init__(config_file)
        self.model_path = self.out_dir + "/dphase_model_final.pt"
        self.best_model_path = self.out_dir + "/dphase_model_best.pt"

        # logging
        self.log_file_main = self.log_dir + 'main.log'
        self.log_file_stats = self.log_dir + 'episodes_stats.csv'
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        file_handler = logging.FileHandler(self.log_file_main, mode='w')
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger_main = logging.getLogger(MAIN_LOG)
        logger_main.setLevel(level=logging.INFO)
        logger_main.addHandler(file_handler)
        logger_main.addHandler(logging.StreamHandler(sys.stdout))
        logger_stats_train = logging.getLogger(STATS_LOG_TRAIN)
        logger_stats_train.setLevel(level=logging.INFO)
        logger_stats_train.info(",".join(STATS_LOG_COLS_TRAIN))
        logger_stats_validate = logging.getLogger(STATS_LOG_VALIDATE)
        logger_stats_validate.setLevel(level=logging.INFO)
        logger_stats_validate.info(",".join(STATS_LOG_COLS_VALIDATE))
        logging.info(self)

    def __str__(self):
        s = super().__str__()
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


class TestConfig(Config):
    def __init__(self, config_file, **entries):
        self.__dict__.update(entries)
        super().__init__(config_file)
        self.phasing_output_path = self.out_dir + "phasing_output.pickle"
        self.output_vcf = self.out_dir + "dphase_phased.vcf"

        # logging
        # noinspection PyArgumentList
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                            handlers=[logging.FileHandler(self.out_dir + '/training.log', mode='w'),
                                      logging.StreamHandler(sys.stdout)])
        logging.info(self)

    def __str__(self):
        s = super().__str__()
        s += '\n\t'.join("{}: {}".format(k, v) for k, v in self.__dict__.items())
        return s


def load_config(fname, config_type=CONFIG_TYPE.TRAIN):
    # Load a YAML configuration file
    with open(fname) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if config_type == CONFIG_TYPE.TRAIN:
        return TrainingConfig(fname, **config)
    elif config_type == CONFIG_TYPE.TEST:
        return TestConfig(fname, **config)



