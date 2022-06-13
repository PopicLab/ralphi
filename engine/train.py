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
#matplotlib.use('macosx')

parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--panel', help='Input fragment files (training data)')
parser.add_argument('--validation_panel', help='Input fragment files (training data)')
parser.add_argument('--pretrained_model', default=None,  help='pretrained model to use at a starting point')
parser.add_argument('--prune_nodes', type=int, default=1, help='Do not train on graphs smaller than prune_nodes (default: 1)')
parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=12345, help='Random seed (default: 12345)')
parser.add_argument('--out_dir', help='Directory for output files (model, plots)')
parser.add_argument('--max_episodes', default=None, type=int, help='Maximum number of episodes to play')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()
torch.manual_seed(args.seed)
random.seed(args.seed)

# Setup the agent and the environment
env = envs.PhasingEnv(args.panel, out_dir=args.out_dir, prune_graphs_smaller_than=args.prune_nodes)
agent = agents.DiscreteActorCriticAgent(env)
validation_agent = agents.DiscreteActorCriticAgent(env)
if args.pretrained_model is not None:
    agent.model.load_state_dict(torch.load(args.pretrained_model))

# Set up some logging to help analyze agent behaviour as we train
csv_stats_storage = open(args.out_dir + "/episodes_stats.csv", 'w')
csv_writer = csv.writer(csv_stats_storage, delimiter=",")
column_headers = ['Episode', 'Nodes', 'Edges', 'Reward', 'CutSize', 'ActorLoss', 'CriticLoss', 'SumLoss', 'Runtime']
csv_writer.writerow(column_headers)

validation_csv_stats_storage = open(args.out_dir + "/validation.csv", 'w')
validation_csv_writer = csv.writer(validation_csv_stats_storage, delimiter=",")
validation_column_headers = ['Episode', 'SumOfCuts', 'SumOfRewards']
validation_csv_writer.writerow(validation_column_headers)

# Play!
sim_mode = False
episode_rewards = []
episode_accuracies = []
graph_num_nodes = []
graph_num_edges = []
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
    cut_size = env.getCutValue()
    print('Episode: {}. Reward: {}, CutSize: {}, Runtime: {}, Accuracy: {}, ActorLoss: {}, CriticLoss: {}, SumLoss: {} '.format(episode, episode_reward, cut_size, end_time - start_time,
          episode_accuracy, actor_loss, critic_loss, sum_loss))
    g_num_nodes = env.state.frag_graph.g.number_of_nodes()
    g_num_edges = env.state.frag_graph.g.number_of_edges()
    graph_num_nodes.append(g_num_nodes)
    graph_num_edges.append(g_num_edges)
    csv_writer.writerow([episode, g_num_nodes, g_num_edges, episode_reward, cut_size, actor_loss, critic_loss, sum_loss, end_time - start_time])
    if episode % 5000 == 0 and args.validation_panel is not None:
        # benchmark the current model against a held out set of fragment graphs (validation panel)
        validation_env = envs.PhasingEnv(panel=args.validation_panel)
        validation_agent.model.load_state_dict(agent.model.state_dict())
        validation_agent.env = validation_env
        validation_sum_of_rewards = 0
        validation_sum_of_cuts = 0
        while validation_env.has_state():
            validation_reward = validation_agent.run_episode_no_updates()
            validation_sum_of_rewards += validation_reward
            validation_sum_of_cuts += validation_env.getCutValue()
            validation_env.reset()
        print("Episode: ", episode, "Sum of Cuts: ", validation_sum_of_cuts, "Sum of Rewards: ", validation_sum_of_rewards)
        validation_csv_writer.writerow([episode, validation_sum_of_cuts, validation_sum_of_rewards])
        validation_csv_stats_storage.flush()  # whenever you want
        if validation_sum_of_cuts > best_sum_of_cuts:
            best_sum_of_cuts = validation_sum_of_cuts
            torch.save(agent.model.state_dict(), "%s/dphase_model_best.pt" % args.out_dir)
        torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (args.out_dir, model_no))
        model_no += 1
    episode += 1
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
