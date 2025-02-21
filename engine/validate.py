from copy import deepcopy
from operator import index

import pandas as pd
import wandb
import logging
import envs.phasing_env as envs
import models.actor_critic as agents
import torch
from joblib import Parallel, delayed

def validation_task(sub_df, config, model_path):
    config = deepcopy(config)
    config.epochs = 1
    logging.root.setLevel(logging.getLevelName(config.logging_level))
    task_component_stats = []
    validate_env = envs.PhasingEnv(config, record_solutions=not config.debug, graph_dataset=sub_df)
    agent = agents.DiscreteActorCriticAgent(validate_env)
    agent.model.load_state_dict(torch.load(model_path))
    index_validate = 0
    while agent.env.has_state():
        sum_of_rewards = 0
        sum_of_cuts = 0
        reward_val = agent.run_episode(test_mode=True)
        sum_of_rewards += reward_val
        cut_val = agent.env.get_cut_value()
        sum_of_cuts += cut_val
        if config.debug:
            cur_index = sub_df.iloc[index_validate, :].values.tolist()
            cur_index.extend([reward_val, cut_val])
            task_component_stats.append(cur_index)
        index_validate += 1
        if index_validate % 100 == 0:
            logging.info('Validation graph %d' % index_validate)
        agent.env.reset()
    return pd.DataFrame(task_component_stats, columns=list(sub_df.columns) + ["reward_val", "cut_val"])

def validate(model_checkpoint_id, episode_id, validation_dataset, config):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    logging.info("running validation with model number: %d, at episode: %d" % (model_checkpoint_id, episode_id))
    model_path = "%s/ralphi_model_%d.pt" % (config.out_dir, model_checkpoint_id)
    validation_component_stats = Parallel(n_jobs=config.n_procs)(delayed(validation_task)(sub_df, config,
                                                                    model_path) for sub_df in validation_dataset)

    validation_indexing_df = pd.concat(validation_component_stats)
    validation_indexing_df.to_pickle("%s/validation_index_for_model_%d.pickle" % (config.out_dir, model_checkpoint_id))
    def log_stats_for_filter(validation_filtered_df):
        metrics_of_interest = ["reward_val", "cut_val"]
        def log_wandb(df, group):
            for metric in metrics_of_interest:
                wandb.log({"Episode": episode_id, "Validation_" + metric + "_" + group: df[metric].sum()})
            wandb.log({"Episode": episode_id, "Validation_number_examples_" + group: len(df)})
        log_wandb(validation_filtered_df, "overall")
        groups = validation_filtered_df['group'].unique()
        for group in groups:
            df_group = validation_filtered_df[validation_filtered_df['group'] == group]
            log_wandb(df_group, group)

    # stats for entire validation set
    log_stats_for_filter(validation_indexing_df)
    return validation_indexing_df["reward_val"].sum()
