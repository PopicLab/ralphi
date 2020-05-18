import argparse
import models.actor_critic as agents
import envs.phasing_env as envs
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import matplotlib
import graphs.haploype_graphs as graphs
matplotlib.use('tkagg')

parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='reward discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=12345, metavar='N',
                    help='random seed (default: 12345)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


# Setup the agent and the environment
env = envs.PhasingEnv()
agent = agents.DiscreteActorCriticAgent(env)

# Play!
sim_mode = True
num_episodes = 200
episode_rewards = []
episode_accuracies = []
for episode in range(num_episodes):
    start_time = time.time()
    episode_reward = agent.run_episode()
    end_time = time.time()
    episode_rewards.append(episode_reward)
    episode_accuracy = 0.0
    if sim_mode:
        node_labels = env.state.g.ndata['x'][:, 0].cpu().squeeze().numpy().tolist()
        episode_accuracy = graphs.eval_assignment(node_labels, env.state.haplotype_graph.node_id2hap_id)
    episode_accuracies.append(100*episode_accuracy)
    print('Episode: {}. Reward: {}, Runtime: {}, Accuracy: {} '.format(episode, episode_reward, end_time - start_time,
          episode_accuracy))

# Plots results
y = np.asarray(episode_rewards)
z = np.asarray(episode_accuracies)
x = np.linspace(0, len(y), len(y))
fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(x, y)
plt.xlabel('Episode')
plt.ylabel('Reward')

fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(x, z)
plt.xlabel('Episode')
plt.ylabel('Accuracy')

plt.show()

# save the model
MODEL_PATH = 'phasing_model.pt'
torch.save(agent.model.state_dict(), MODEL_PATH)
