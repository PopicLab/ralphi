import argparse
import pandas as pd
import graphs.frag_graph
import models.actor_critic as agents
import envs.phasing_env as envs
import torch
import random
import engine.config as config_utils
import engine.validate
from concurrent.futures import ThreadPoolExecutor

# ------ CLI ------
parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--config', help='Training configuration YAML')
parser.add_argument('--training_dataset_config', help='Config to specify training data design')
parser.add_argument('--validation_dataset_config', help='Config to specify validation data design')
args = parser.parse_args()
# -----------------

# Load the config
config = config_utils.load_config(args.config)
training_config = config_utils.load_config(args.training_dataset_config, config_type=config_utils.CONFIG_TYPE.DATA_DESIGN)
validation_config = config_utils.load_config(args.validation_dataset_config, config_type=config_utils.CONFIG_TYPE.DATA_DESIGN)

torch.manual_seed(config.seed)
random.seed(config.seed)
torch.set_num_threads(config.num_cores)

training_dataset = graphs.frag_graph.GraphDataset(config, training_config, validation_mode=False).load_indices()

if config.panel_validation_frags and config.panel_validation_vcfs:
    validation_dataset = graphs.frag_graph.GraphDataset(config, validation_config, validation_mode=True).load_indices()
    # e.g. to only validate on cases with articulation points
    # validation_dataset = validation_dataset[validation_dataset["articulation_points"] != 0]

# Setup the agent and the training environment
env_train = envs.PhasingEnv(config, graph_dataset=training_dataset)
agent = agents.DiscreteActorCriticAgent(env_train)
if config.pretrained_model is not None:
    agent.model.load_state_dict(torch.load(config.pretrained_model))

# Run the training
best_validation_reward = 0
model_checkpoint_id = 0
episode_id = 0

with ThreadPoolExecutor(max_workers=config.validation_parallel_chunks) as executor:
    while agent.env.has_state():
        if config.max_episodes is not None and episode_id >= config.max_episodes:
            break
        episode_reward = agent.run_episode(config, episode_id=episode_id)
        if episode_id % config.interval_validate == 0:
            torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
            if config.panel_validation_frags and config.panel_validation_vcfs:
                reward = engine.validate.validate(model_checkpoint_id, episode_id, validation_dataset, agent, config, executor)
                model_checkpoint_id += 1
                if reward > best_validation_reward:
                    best_validation_reward = reward
                    torch.save(agent.model.state_dict(), config.best_model_path)
        episode_id += 1
        agent.env = env_train
        agent.env.reset()

# save the model
torch.save(agent.model.state_dict(), config.model_path)
