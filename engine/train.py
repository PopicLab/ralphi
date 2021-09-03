import argparse
import models.actor_critic as agents
import envs.phasing_env as envs
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import os.path
import matplotlib
import graphs.frag_graph as graphs
#matplotlib.use('macosx')

parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--panel', default="../data/train/frags/chr20/panel_gs_chr20_final_local.txt",
                    help='Input fragment files (training data)')
parser.add_argument('--uid', default="e_all_all", help='Experiment uid')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='reward discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=12345, metavar='N',
                    help='random seed (default: 12345)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
args = parser.parse_args()

# Setup the agent and the environment
env = envs.PhasingEnv(args.panel)
agent = agents.DiscreteActorCriticAgent(env)

# Bookkeeping
#experiment_name = os.path.splitext(os.path.basename(args.panel))[0]
experiment_name = args.uid
max_episodes = None
episode = 0

# Play!
sim_mode = False
episode_rewards = []
episode_accuracies = []
while env.has_state():
    if max_episodes is not None and episode >= max_episodes:
        break
    start_time = time.time()
    episode_reward = agent.run_episode()
    end_time = time.time()
    episode_rewards.append(episode_reward)
    episode_accuracy = 0.0
    if sim_mode:
        node_labels = env.state.g.ndata['x'][:, 0].cpu().squeeze().numpy().tolist()
        episode_accuracy = graphs.eval_assignment(node_labels, env.state.frag_graph.node_id2hap_id)
    episode_accuracies.append(100*episode_accuracy)
    print('Episode: {}. Reward: {}, Runtime: {}, Accuracy: {} '.format(episode, episode_reward, end_time - start_time,
          episode_accuracy))
    episode += 1
    env.reset()

# save the model
MODEL_PATH = '../data/train/models/' + experiment_name + '_phasing_model.pt'
torch.save(agent.model.state_dict(), MODEL_PATH)

# Plots results
y = np.asarray(episode_rewards)
z = np.asarray(episode_accuracies)
x = np.linspace(0, len(y), len(y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, y)
plt.xlabel('Episode')
plt.ylabel('Reward')
fig2.savefig('../data/train/results/reward.png')

fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x, z)
plt.xlabel('Episode')
plt.ylabel('Accuracy')
fig3.savefig('../data/train/results/accuracy.png')

#plt.show()
