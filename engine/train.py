import argparse
import models.actor_critic as agents
import envs.phasing_env as envs
import torch
import random
import graphs.frag_graph as graphs
import engine.config as config_utils
import logging

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

# Setup the agent and the environment
agent = agents.DiscreteActorCriticAgent(
    envs.PhasingEnv(config.panel_train, out_dir=config.out_dir,
                    min_graph_size=config.min_graph_size,
                    max_graph_size=config.max_graph_size,
                    skip_trivial_graphs=True))
if config.pretrained_model is not None:
    agent.model.load_state_dict(torch.load(config.pretrained_model))

def validate(current_best_reward):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    # TODO: pre-load the validation panel
    agent.env = envs.PhasingEnv(panel=config.panel_validate, skip_trivial_graphs=True)
    sum_of_rewards = 0
    sum_of_cuts = 0
    while agent.env.has_state():
        sum_of_rewards += agent.run_episode(test_mode=True)
        sum_of_cuts += agent.env.get_cut_value()
        agent.env.reset()
    torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
    # log episode stats
    logging.getLogger(config.MAIN_LOG).info("Episode: %d, Sum of Cuts: %d, Sum of Rewards: %d" %
                                            (episode_id, sum_of_cuts, sum_of_rewards))
    logging.getLogger(config.STATS_LOG_VALIDATE).info(",".join([episode_id, sum_of_cuts, sum_of_rewards]))
    return sum_of_rewards

# Run the training
sim_mode = False
episode_accuracies = []
best_validation_reward: int = 0
model_checkpoint_id: int = 0
episode_id = 0
while agent.env.has_state():
    if config.max_episodes is not None and episode_id >= config.max_episodes:
        break
    episode_reward = agent.run_episode(episode_id=episode_id)
    if episode_id % config.interval_validate == 0 and config.panel_validate is not None:
        reward = validate()
        model_checkpoint_id += 1
        if reward > best_validation_reward:
            best_validation_reward = reward
            torch.save(agent.model.state_dict(), config.best_model_path)
    episode_id += 1
    if sim_mode:
        node_labels = agent.env.state.g.ndata['x'][:, 0].cpu().squeeze().numpy().tolist()
        episode_accuracy = graphs.eval_assignment(node_labels, agent.env.state.frag_graph.node_id2hap_id)
        episode_accuracies.append(100*episode_accuracy)
    if config.render:
        agent.env.render('weighted_view')
    agent.env.reset()

# save the model
torch.save(agent.model.state_dict(), config.model_path)
