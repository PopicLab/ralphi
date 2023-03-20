import argparse
import pandas as pd
import graphs.frag_graph
import models.actor_critic as agents
import envs.phasing_env as envs
import torch
import random
import engine.config as config_utils
import engine.validate
import logging
import wandb
import seq.var as var
import seq.phased_vcf as vcf_writer
import utils.post_processing
import os
import third_party.HapCUT2.utilities.calculate_haplotype_statistics as benchmark
import sys
import utils.hap_block_visualizer as hap_block_visualizer
import pickle
import tqdm

# ------ CLI ------
parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--config', help='Training configuration YAML')
args = parser.parse_args()
# -----------------

# Load the config
config = config_utils.load_config(args.config)

# logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                            handlers=[logging.FileHandler(config.log_dir + '/training.log', mode='w'),
                                                              logging.StreamHandler(sys.stdout)])

torch.manual_seed(config.seed)
random.seed(config.seed)
torch.set_num_threads(config.num_cores)

# set up performance tracking
if config.log_wandb:
    wandb.init(project="dphase_experiments", entity="dphase", dir=config.log_dir)
else:
    # automatically results in ignoring all wandb calls
    wandb.init(project="dphase_experiments", entity="dphase", dir=config.log_dir, mode="disabled")


training_distribution = graphs.frag_graph.GraphDataset(config, load_components=True, store_components=True, save_indexes=True, validation_mode=False)
training_dataset = training_distribution.load_graph_dataset_indices()

if config.panel_validation_frags and config.panel_validation_vcfs:
    validation_distribution = graphs.frag_graph.GraphDataset(config, load_components=True, store_components=True, save_indexes=True, validation_mode=True)
    validation_dataset = validation_distribution.load_graph_dataset_indices()
    # e.g. to only validate on cases with articulation points
    # validation_dataset = validation_dataset[validation_dataset["articulation_points"] != 0]


# Setup the agent and the training environment
env_train = envs.PhasingEnv(config.panel_train,
                            min_graph_size=config.min_graph_size,
                            max_graph_size=config.max_graph_size,
                            skip_trivial_graphs=config.skip_trivial_graphs, graph_distribution=training_dataset, compress=config.compress)
agent = agents.DiscreteActorCriticAgent(env_train)
if config.pretrained_model is not None:
    agent.model.load_state_dict(torch.load(config.pretrained_model))

# Run the training
best_validation_reward = 0
model_checkpoint_id = 0
episode_id = 0
while agent.env.has_state():
    if config.max_episodes is not None and episode_id >= config.max_episodes:
        break
    episode_reward = agent.run_episode(config, episode_id=episode_id)
    if episode_id % config.interval_validate == 0:
        torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
        if config.panel_validation_frags and config.panel_validation_vcfs:
            reward = engine.validate.validate(model_checkpoint_id, episode_id, validation_dataset, agent, config)
            model_checkpoint_id += 1
            if reward > best_validation_reward:
                best_validation_reward = reward
                torch.save(agent.model.state_dict(), config.best_model_path)
    episode_id += 1
    agent.env = env_train
    agent.env.reset()

# save the model
torch.save(agent.model.state_dict(), config.model_path)
