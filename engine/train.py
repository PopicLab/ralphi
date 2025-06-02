import argparse
import logging
import torch
import random
import dgl

from graphs.graph_dataset import GraphDataset
import models.actor_critic as agents
import envs.phasing_env as envs
import engine.config as config_utils
import engine.validate


if __name__ == '__main__':
    # ------ CLI ------
    parser = argparse.ArgumentParser(description='Train haplotype phasing')
    parser.add_argument('--config', help='Training configuration YAML')
    args = parser.parse_args()
    # -----------------
    # Load the config
    config = config_utils.load_config(args.config)
    if not config.panel_dataset_train or not config.panel_dataset_validate:
        raise FileNotFoundError('Datasets files for training and validation are required. '
                                'Please use design_dataset.py to generate them.')
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    dgl.seed(config.seed)
    torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_num_threads(config.num_cores_torch)
    # Load Datasets
    training_dataset = GraphDataset.load_dataframe(config.panel_dataset_train)
    validation_dataset = GraphDataset.load_dataframe(config.panel_dataset_validate)
    validation_dataset = GraphDataset.round_robin(config.n_procs, validation_dataset)

    # Setup the agent and the training environment
    env_train = envs.PhasingEnv(config, graph_dataset=training_dataset)
    agent = agents.DiscreteActorCriticAgent(env_train)
    if config.pretrained_model is not None:
        agent.model.load_state_dict(torch.load(config.pretrained_model))

    # Run the training
    best_validation_reward = 0
    best_validation_reward_id = 0
    model_checkpoint_id = 0
    episode_id = 0

    while agent.env.has_state():
        if config.max_episodes is not None and episode_id >= config.max_episodes:
            break
        reward = None
        if episode_id % config.interval_validate == 0:
            torch.save(agent.model.state_dict(), "%s/ralphi_model_%d.pt" % (config.out_dir, model_checkpoint_id))
            reward = engine.validate.validate(model_checkpoint_id, episode_id, validation_dataset, config)
            model_checkpoint_id += 1
            if reward > best_validation_reward:
                best_validation_reward = reward
                best_validation_reward_id = model_checkpoint_id
                torch.save(agent.model.state_dict(), config.best_model_path)
        episode_reward = agent.run_episode(config, episode_id=episode_id)
        episode_id += 1
        agent.env = env_train
        if episode_id % (10 * config.interval_validate) == 0:
            logging.info('Finished Episode %s, obtained a reward of %s at validation step %s' % (episode_id, reward, model_checkpoint_id))
            logging.info('Best validation reward obtained is %d at validation step %d' % (best_validation_reward, best_validation_reward_id))
        agent.env.reset()

    # save the model
    torch.save(agent.model.state_dict(), config.model_path)
