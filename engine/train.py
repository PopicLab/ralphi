import argparse
import models.actor_critic as agents
import envs.phasing_env as envs
import torch
import random
import engine.config as config_utils
import logging
import wandb

# ------ CLI ------
parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--config', help='Training configuration YAML')
args = parser.parse_args()
# -----------------

# Load the config
config = config_utils.load_config(args.config)
torch.manual_seed(config.seed)
random.seed(config.seed)
torch.set_num_threads(config.num_cores)

# set up performance tracking
if config.debug:
    wandb.init(project="debugging", entity="dphase")


# Setup the agent and the training environment
env_train = envs.PhasingEnv(config.panel_train,
                            min_graph_size=config.min_graph_size,
                            max_graph_size=config.max_graph_size,
                            skip_trivial_graphs=config.skip_trivial_graphs)
agent = agents.DiscreteActorCriticAgent(env_train)
if config.pretrained_model is not None:
    agent.model.load_state_dict(torch.load(config.pretrained_model))

def validate(model_checkpoint_id):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    # TODO: pre-load the validation panel
    agent.env = envs.PhasingEnv(panel=config.panel_validate, skip_trivial_graphs=config.skip_trivial_graphs, debug=True, record_solutions=True)
    sum_of_rewards = 0
    sum_of_cuts = 0
    episode_no = 0
    if config.debug:
        # ✨ W&B: Create a Table to store predictions for each test step
        columns=["id", "number of nodes", "number of edges", "density", "radius", "diameter", "number of variants", "cut value"]
        test_table = wandb.Table(columns=columns)
    while agent.env.has_state():
        sum_of_rewards += agent.run_episode(config, test_mode=True)
        sum_of_cuts += agent.env.get_cut_value()
        if config.debug:
            graph_stats = agent.env.get_graph_stats()
            test_table.add_data(episode_no, graph_stats["num_nodes"], graph_stats["num_edges"], graph_stats["density"], graph_stats["radius"], graph_stats["diameter"], graph_stats["n_variants"], graph_stats["cut_value"])
        agent.env.reset()
        episode_no += 1

    torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
    # log validation loop stats
    logging.getLogger(config_utils.MAIN_LOG).info("Validation checkpoint: %d, Sum of Cuts: %d, Sum of Rewards: %d" %
                                                  (model_checkpoint_id, sum_of_cuts, sum_of_rewards))
    logging.getLogger(config_utils.STATS_LOG_VALIDATE).info("%s,%s,%s" % (episode_id, sum_of_cuts, sum_of_rewards))
    if config.debug:
        wandb.log({"Validation Sum of Rewards": sum_of_rewards})
        wandb.log({"Validation Sum of Cuts": sum_of_cuts})
        wandb.log({"validation_predictions_" + str(model_checkpoint_id): test_table})
    return sum_of_rewards


# Run the training
best_validation_reward = 0
model_checkpoint_id = 0
episode_id = 0
while agent.env.has_state():
    if config.max_episodes is not None and episode_id >= config.max_episodes:
        break
    episode_reward = agent.run_episode(config, episode_id=episode_id)
    if episode_id % config.interval_validate == 0 and config.panel_validate is not None:
        reward = validate(model_checkpoint_id)
        model_checkpoint_id += 1
        if reward > best_validation_reward:
            best_validation_reward = reward
            torch.save(agent.model.state_dict(), config.best_model_path)
    episode_id += 1
    agent.env = env_train
    agent.env.reset()

# save the model
torch.save(agent.model.state_dict(), config.model_path)
