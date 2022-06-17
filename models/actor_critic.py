import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_models import GCN, GCNFirstLayer
import numpy as np


class ActorCriticNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_actions):
        super(ActorCriticNet, self).__init__()
        # linear transformation will be applied to the last dimension of the input tensor
        # which must equal hidden_dim -- number of features per node
        self.policy_graph = nn.Linear(hidden_dim, 1)
        self.policy_done = nn.Linear(hidden_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])

        self.actions = []
        self.rewards = []

    def forward(self, genome_graph):
        """
        Returns:
        - actor's policy: tensor of probabilities for each action
        - critic's value of the current state
        """
        #h = torch.cat([genome_graph.ndata['x'], genome_graph.ndata['y'].float()], dim=1)
        h = genome_graph.ndata['x']
        for conv in self.layers:
            h = conv(genome_graph, h)
        genome_graph.ndata['h'] = h
        mN = dgl.mean_nodes(genome_graph, 'h')
        v = self.value(mN)
        pi = self.policy_graph(genome_graph.ndata['h'])
        pi_done = self.policy_done(mN)
        pi = torch.cat([pi, pi_done])
        genome_graph.ndata.pop('h')
        return pi, v


class DiscreteActorCriticAgent:
    def __init__(self, env):
        self.env = env
        self.model = ActorCriticNet(1, 264, self.env.num_actions)
        self.gamma = 0.98
        # very small learning rate appears to stabilize training; TODO (Anant): experiment with LR scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000003)
        self.batch_size = 32

    def select_action(self, greedy=False):
        if self.env.state.num_nodes < 2:
            return 0
        [pi, val] = self.model(self.env.state.g)
        pi = pi.squeeze()
        pi[self.env.get_all_invalid_actions()] = -float('Inf')
        if greedy:
            greedy_choice = torch.argmax(pi)
            return greedy_choice.item()
        pi = F.softmax(pi, dim=0)
        # print("pi: ", pi)
        dist = torch.distributions.categorical.Categorical(pi)
        action = dist.sample()
        # print("action: ", action)
        self.model.actions.append((dist.log_prob(action), val[0]))
        return action.item()

    def run_episode(self, greedy=False):
        print("Run episode.....")
        #self.env.reset()
        if self.env.state.num_nodes < 2:
            return 0
        done = False
        episode_reward = 0
        while not done:
            action = self.select_action(greedy)
            _, reward, done = self.env.step(action)
            self.model.rewards.append(reward)
            episode_reward += reward
            #print("Action: ", action, "reward", reward, "is_done", done)
        if greedy:
            del self.model.rewards[:]
            del self.model.actions[:]
            # self.env.render()
            return episode_reward
        (actor_loss, critic_loss, sum_loss) = self.update_model()
        # self.env.render()
        return episode_reward, actor_loss, critic_loss, sum_loss

    def update_model(self):
        R = 0
        rewards = []
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        # returns = (returns - returns.mean()) / (returns.std() + eps)
        loss_policy = []
        loss_value = []
        for (log_prob, value), reward in zip(self.model.actions, rewards):
            advantage = reward - value.item()
            loss_policy.append(-log_prob * advantage)
            loss_value.append(F.smooth_l1_loss(value, torch.tensor([reward])))
        # reset gradients
        self.optimizer.zero_grad()
        loss = torch.stack(loss_policy).sum() + torch.stack(loss_value).sum()
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.actions[:]
        return (torch.stack(loss_policy).sum().item(), torch.stack(loss_value).sum().item(), loss.item())
