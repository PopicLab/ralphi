import argparse
import models.actor_critic as agents
import envs.phasing_env as envs
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import os.path
import matplotlib
import csv
import random
import graphs.frag_graph as graphs
import utils.logging
#matplotlib.use('macosx')

parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--panel', help='Input fragment files (training data)')
parser.add_argument('--validation_panel', help='Input fragment files (training data)')
parser.add_argument('--pretrained_model', default=None,  help='pretrained model to use at a starting point')
parser.add_argument('--min_graph_size', type=int, default=1, help='Do not train on graphs smaller than min_graph_size (default: 1)')
parser.add_argument('--max_graph_size', type=int, default=float('inf'), help='Do not train on graphs larger than max_graph_size (default: inf)')
parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=12345, help='Random seed (default: 12345)')
parser.add_argument('--out_dir', help='Directory for output files (model, plots)')
parser.add_argument('--max_episodes', default=None, type=int, help='Maximum number of episodes to play')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--num_cores', type=int, default=4, help='number of threads to use for Pytorch (default: 4)')

args = parser.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)
torch.set_num_threads(args.num_cores)

def benchmarking_training_loop(model_no, current_best):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    validation_env = envs.PhasingEnv(panel=args.validation_panel, skip_trivial_graphs=True)
    validation_agent.model.load_state_dict(agent.model.state_dict())
    validation_agent.env = validation_env
    validation_sum_of_rewards = 0
    validation_sum_of_cuts = 0
    while validation_env.has_state():
        validation_reward = validation_agent.run_episode(greedy=True)
        validation_sum_of_rewards += validation_reward
        validation_sum_of_cuts += validation_env.get_cut_value()
        validation_env.reset()
    print("Episode: ", episode, "Sum of Cuts: ", validation_sum_of_cuts, "Sum of Rewards: ", validation_sum_of_rewards)
    logger.write_validation_stats(episode, validation_sum_of_cuts, validation_sum_of_rewards)
    if validation_sum_of_cuts > current_best:
        current_best = validation_sum_of_cuts
        torch.save(agent.model.state_dict(), "%s/dphase_model_best.pt" % args.out_dir)
    torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (args.out_dir, model_no))
    return current_best


# Setup the agent and the environment
env = envs.PhasingEnv(args.panel, out_dir=args.out_dir, min_graph_size=args.min_graph_size,
                      max_graph_size=args.max_graph_size, skip_trivial_graphs=True)
agent = agents.DiscreteActorCriticAgent(env)
validation_agent = agents.DiscreteActorCriticAgent(env)
if args.pretrained_model is not None:
    agent.model.load_state_dict(torch.load(args.pretrained_model))

logger = utils.logging.Logger(args.out_dir)

# Play!
sim_mode = False
episode_rewards = []
episode_accuracies = []
episode = 0
model_no = 0
best_sum_of_cuts = 0
while env.has_state():
    if args.max_episodes is not None and episode >= args.max_episodes:
        break
    start_time = time.time()
    episode_reward, actor_loss, critic_loss, sum_loss = agent.run_episode()
    end_time = time.time()
    episode_rewards.append(episode_reward)
    episode_accuracy = 0.0
    if sim_mode:
        node_labels = env.state.g.ndata['x'][:, 0].cpu().squeeze().numpy().tolist()
        episode_accuracy = graphs.eval_assignment(node_labels, env.state.frag_graph.node_id2hap_id)
    episode_accuracies.append(100*episode_accuracy)

    cut_size, g_num_nodes, g_num_edges = env.get_graph_stats()

    print('Episode: {}. Reward: {}, CutSize: {}, Runtime: {}, Accuracy: {}, ActorLoss: {}, CriticLoss: {}, SumLoss: {} '.format(episode, episode_reward, cut_size, end_time - start_time,
          episode_accuracy, actor_loss, critic_loss, sum_loss))

    logger.write_episode_stats(episode, g_num_nodes, g_num_edges, episode_reward, cut_size, actor_loss, critic_loss, sum_loss, end_time - start_time)

    if episode % 5000 == 0 and args.validation_panel is not None:
        best_sum_of_cuts = benchmarking_training_loop(model_no, best_sum_of_cuts)
        model_no += 1

    episode += 1
    if args.render:
        env.render('weighted_view')
    env.reset()

# save the model
torch.save(agent.model.state_dict(), "%s/dphase_model_final.pt" % args.out_dir)

# Plots results
y = np.asarray(episode_rewards)
z = np.asarray(episode_accuracies)
x = np.linspace(0, len(y), len(y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, y)
plt.xlabel('Episode')
plt.ylabel('Reward')
fig2.savefig("%s/reward.png" % args.out_dir)

fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x, z)
plt.xlabel('Episode')
plt.ylabel('Accuracy')
fig3.savefig("%s/accuracy.png" % args.out_dir)
#plt.show()
