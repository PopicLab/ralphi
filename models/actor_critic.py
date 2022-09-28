import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_models import GCN, GCNFirstLayer
import time
import logging
import engine.config as config
import engine.constants as constants

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

    def run_episode(self, config, test_mode=False, episode_id=None):
        start_time = time.time()
        if self.env.state.num_nodes < 2:
            return 0
        done = False
        episode_reward = 0
        while not done:
            action = self.select_action(test_mode)
            _, reward, done = self.env.step(action)
            episode_reward += reward
            if not test_mode:
                self.model.rewards.append(reward)
        if not test_mode:
            loss = self.update_model()
            self.log_episode_stats(episode_id, episode_reward, loss, time.time() - start_time)
        if config.render:
            self.env.render(config.render_view)
        return episode_reward

    def log_episode_stats(self, episode_id, reward, loss, runtime):
        graph_stats = self.env.get_graph_stats()
        logging.getLogger(config.MAIN_LOG).info("Episode: %d. Reward: %d, ActorLoss: %d, CriticLoss: %d, TotalLoss: %d,"
                                                " CutSize: %d, Runtime: %d" %
                                                (episode_id, reward,
                                                 loss[constants.LossTypes.actor_loss],
                                                 loss[constants.LossTypes.critic_loss],
                                                 loss[constants.LossTypes.total_loss],
                                                 graph_stats[constants.GraphStats.cut_value],
                                                 runtime))
        logging.getLogger(config.STATS_LOG_TRAIN).info(",".join([str(episode_id), str(reward),
                                                                 ",".join(str(loss) for loss in loss.values()),
                                                                 ",".join(str(stat) for stat in graph_stats.values()),
                                                                 str(runtime)]))

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
        return {
           constants.LossTypes.actor_loss: torch.stack(loss_policy).sum().item(),
           constants.LossTypes.critic_loss: torch.stack(loss_value).sum().item(),
           constants.LossTypes.total_loss: loss.item()
        }

