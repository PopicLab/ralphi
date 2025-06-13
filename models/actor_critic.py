import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from models.graph_models import GIN, GCN, GCNv2
import time
import logging
import engine.config as config
import models.constants as constants
import wandb
from torch.nn.utils import spectral_norm


class ActorCriticNet(nn.Module):
    def __init__(self, config):
        layers_dict = {"gin": GIN, "gcn": GCN, "gcn2": GCNv2}
        super(ActorCriticNet, self).__init__()
        self.policy_graph_hap0 = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.policy_graph_hap0.weight.data)
        self.policy_graph_hap1 = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.policy_graph_hap1.weight.data)
        self.policy_done = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.policy_done.weight.data)
        self.value = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.value.weight.data)
        self.layers = layers_dict[config.layer_type](config.node_features_dim,
                                                     config.hidden_dim, **config.embedding_vars)
        self.feature_list = list(feature for feature_name in config.features
                             for feature in constants.FEATURES_DICT[feature_name])

        self.actions = []
        self.rewards = []

    def forward(self, genome_graph):
        """
        Returns:
        - actor's policy: tensor of probabilities for each action
        - critic's value of the current state
        """
        features = list(genome_graph.ndata[feature] for feature in self.feature_list)
        weights = genome_graph.edata['weight']
        graph_emb = torch.cat(features, dim=1)
        graph_emb = self.layers(genome_graph, graph_emb, edge_feat=weights[:, None],
                                edge_weights=weights, etypes=torch.gt(weights, 0))[-1]
        genome_graph.ndata['graph_emb'] = graph_emb
        nodes_mean = dgl.mean_nodes(genome_graph, 'graph_emb')
        pi_hap0 = self.policy_graph_hap0(genome_graph.ndata['graph_emb'])
        pi_hap1 = self.policy_graph_hap1(genome_graph.ndata['graph_emb'])
        pi = torch.cat([pi_hap0, pi_hap1])
        genome_graph.ndata.pop('graph_emb')
        return pi, self.value(nodes_mean)


class DiscreteActorCriticAgent:
    def __init__(self, env):
        self.env = env
        self.model = ActorCriticNet(self.env.config).to(self.env.config.device)
        self.learning_mode = False

    def set_learning_params(self):
        self.gamma = self.env.config.gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.env.config.lr)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.env.config.device)

    def run_episode(self, config, test_mode=False, episode_id=None, group='Global'):
        done = False
        episode_reward = 0
        first = True
        while not done:
            action = self.select_action(test_mode, first=first)
            first = False
            _, reward, done = self.env.step(action)
            episode_reward += reward
            if not test_mode: self.model.rewards.append(reward)
        if not test_mode:
            groups = {'Global'}
            groups.add(group)
            self.update_model(episode_id, groups=groups)
            cut_size = self.env.get_cut_value()
            for group in groups:
                wandb.log({"Episode": episode_id, "Training Episode Reward " + group: episode_reward})
                wandb.log({"Episode": episode_id, "Training Cut Size " + group: cut_size})
        return episode_reward

    def log_episode_stats(self, episode_id, reward, loss, runtime):
        self.env.state.frag_graph.log_graph_properties(episode_id)
        graph_stats = []
        logging.getLogger(config.MAIN_LOG).info("Episode: %d. Reward: %d, ActorLoss: %d, CriticLoss: %d, TotalLoss: %d,"
                                                " CutSize: %d, Runtime: %d" %
                                                (episode_id, reward,
                                                 loss[constants.LossTypes.actor_loss],
                                                 loss[constants.LossTypes.critic_loss],
                                                 loss[constants.LossTypes.total_loss],
                                                 self.env.get_cut_value(),
                                                 runtime))
        logging.getLogger(config.STATS_LOG_TRAIN).info(",".join([str(episode_id), str(reward),
                                                                 str(self.env.get_cut_value()),
                                                                 str(loss),
                                                                 str(graph_stats),
                                                                 str(runtime)]))

    def select_action(self, greedy=False, first=False):
        # based on DAC model in:
        # https://github.com/orrivlin/MinimumVertexCover_DRL/blob/master/discrete_actor_critic.py
        if self.env.state.num_nodes < 2:
            return 0
        [pi, val] = self.model(self.env.state.g)
        pi = pi.squeeze()
        pi[self.env.get_all_invalid_actions()] = -float('Inf')
        if greedy: return torch.argmax(pi).item()
        dist = Categorical(F.softmax(pi, dim=0))
        action = dist.sample()
        self.model.actions.append((dist.log_prob(action), val[0]))
        return action.item()

    def update_model(self, episode_id=None, groups=None):
        if not self.learning_mode:
            self.set_learning_params()
            self.learning_mode = True
        R = 0
        rewards = []
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards).to(self.env.config.device)
        loss_policy = []
        loss_value = []
        for (log_prob, value), reward in zip(self.model.actions, rewards):
            advantage = reward - value.item()
            loss_policy.append(-log_prob * advantage)
            loss_value.append(F.smooth_l1_loss(value, torch.tensor([reward]).to(self.env.config.device)))
        # reset gradients
        self.optimizer.zero_grad()
        loss = torch.stack(loss_policy).sum() + torch.stack(loss_value).sum()
        for group in groups:
            wandb.log({"Episode": episode_id, "loss " + group: loss})
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.actions[:]
        for group in groups:
            wandb.log({"Episode": episode_id, "actor loss " + group: torch.stack(loss_policy).sum().item()})
            wandb.log({"Episode": episode_id, "critic loss " + group: torch.stack(loss_value).sum().item()})

        return {
            constants.LossTypes.actor_loss: torch.stack(loss_policy).sum().item(),
            constants.LossTypes.critic_loss: torch.stack(loss_value).sum().item(),
            constants.LossTypes.total_loss: loss.item()
        }