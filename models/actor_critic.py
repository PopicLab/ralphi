import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.graph_models import GAT, GINE, GIN, PNA, GCN, GCNv2
import time
import logging
import engine.config as config
import models.constants as constants
import wandb
from torch.nn.utils.parametrizations import spectral_norm

class ActorCriticNet(nn.Module):
    def __init__(self, config):
        layers_dict = {"gat": GAT, "gine": GINE, "gin": GIN, "pna": PNA, "gcn": GCN, "gcn2": GCNv2}
        super(ActorCriticNet, self).__init__()
        # linear transformation will be applied to the last dimension of the input tensor
        # which must equal hidden_dim -- number of features per node
        self.policy_graph_hap0 = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.policy_graph_hap0.weight.data)
        self.policy_graph_hap1 = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.policy_graph_hap1.weight.data)
        self.policy_done = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.policy_done.weight.data)
        self.value = spectral_norm(nn.Linear(config.hidden_dim[-1], 1))
        nn.init.orthogonal_(self.value.weight.data)
        self.layers = layers_dict[config.layer_type](config.in_dim, config.hidden_dim, **config.embedding_vars)


        self.actions = []
        self.rewards = []

    def forward(self, genome_graph):
        """
        Returns:
        - actor's policy: tensor of probabilities for each action
        - critic's value of the current state
        """
        features = list(genome_graph.ndata[elem.value] for elem in constants.NodeFeatures)
        h = torch.cat(features, dim=1)
        weights = genome_graph.edata['weight']
        h = self.layers(genome_graph, h, edge_feat=weights[:, None], edge_weights=weights, etypes=torch.gt(weights,0))[-1]

        genome_graph.ndata['h'] = h
        mN = dgl.mean_nodes(genome_graph, 'h')
        v = self.value(mN)
        pi_hap0 = self.policy_graph_hap0(genome_graph.ndata['h'])
        pi_hap1 = self.policy_graph_hap1(genome_graph.ndata['h'])
        pi = torch.cat([pi_hap0, pi_hap1])
        genome_graph.ndata.pop('h')
        return pi, v


class DiscreteActorCriticAgent:
    def __init__(self, env):
        self.env = env
        self.model = ActorCriticNet(self.env.config)
        self.learning_mode = False

    def set_learning_params(self):
        self.gamma = self.env.config.gamma
        # very small learning rate appears to stabilize training; TODO (Anant): experiment with LR scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.env.config.lr)
        self.batch_size = 32

    def select_action(self, greedy=False, first=False):
        if self.env.state.num_nodes < 2:
            return 0
        [pi, val] = self.model(self.env.state.g)
        pi = pi.squeeze()
        """if not first:
            pi[self.env.get_all_non_neighbour_actions()] = -float('Inf')"""
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
        done = False
        episode_reward = 0
        first = True
        while not done:
            action = self.select_action(test_mode, first=first)
            first = False
            _, reward, done = self.env.step(action)
            episode_reward += reward
            if not test_mode:
                self.model.rewards.append(reward)
        if not test_mode:
            loss = self.update_model(episode_id)
            cut_size = self.env.get_cut_value()
            self.log_episode_stats(episode_id, episode_reward, loss, time.time() - start_time)
            wandb.log({"Episode": episode_id, "Training Episode Reward": episode_reward})
            wandb.log({"Episode": episode_id, "Training Cut Size": cut_size})
        if config.render:
            self.env.render(config.render_view)
        return episode_reward

    def log_episode_stats(self, episode_id, reward, loss, runtime):
        self.env.state.frag_graph.log_graph_properties(episode_id)
        graph_stats = self.env.state.frag_graph.graph_properties

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

    def update_model(self, episode_id=None):
        if not self.learning_mode:
            self.set_learning_params()
            self.learning_mode = True
        R = 0
        rewards = []
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        loss_policy = []
        loss_value = []
        for (log_prob, value), reward in zip(self.model.actions, rewards):
            advantage = reward - value.item()
            loss_policy.append(-log_prob * advantage)
            loss_value.append(F.smooth_l1_loss(value, torch.tensor([reward])))
        # reset gradients
        self.optimizer.zero_grad()
        loss = torch.stack(loss_policy).sum() + torch.stack(loss_value).sum()
        wandb.log({"Episode": episode_id, "loss": loss})
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.actions[:]
        wandb.log({"Episode": episode_id, "actor loss": torch.stack(loss_policy).sum().item()})
        wandb.log({"Episode": episode_id, "critic loss": torch.stack(loss_value).sum().item()})
        return {
            constants.LossTypes.actor_loss: torch.stack(loss_policy).sum().item(),
            constants.LossTypes.critic_loss: torch.stack(loss_value).sum().item(),
            constants.LossTypes.total_loss: loss.item()
        }
