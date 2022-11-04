import argparse
import dataset_gen.graph_generator
import models.actor_critic as agents
import envs.phasing_env as envs
import torch
import random
import engine.config as config_utils
import logging
import wandb
import seq.var as var
import seq.phased_vcf as vcf_writer
import utils.post_processing
import os
import third_party.HapCUT2.utilities.calculate_haplotype_statistics as benchmark
import sys

# ------ CLI ------
parser = argparse.ArgumentParser(description='Train haplotype phasing')
parser.add_argument('--config', help='Training configuration YAML')
args = parser.parse_args()
# -----------------

# Load the config
config = config_utils.load_config(args.config)

# logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO,
                            handlers=[logging.FileHandler(config.log_dir + '/training.log', mode='w'),
                                                              logging.StreamHandler(sys.stdout)])

torch.manual_seed(config.seed)
random.seed(config.seed)
torch.set_num_threads(config.num_cores)

if os.path.exists(config.out_dir + "/benchmark.txt"):
    os.remove(config.out_dir + "/benchmark.txt")

# set up performance tracking
if config.debug:
    wandb.init(project="debugging", entity="dphase", dir=config.log_dir)
else:
    # automatically results in ignoring all wandb calls
    wandb.init(project="debugging", entity="dphase", dir=config.log_dir, mode="disabled")


graph_dataset_indices = None
if config.define_training_distribution:
    training_distribution = dataset_gen.graph_generator.TrainingDistribution(config.panel_train, load_components=True, store_components=True, save_indexes=True)
    graph_dataset_indices = training_distribution.load_graph_dataset_indices()


# Setup the agent and the training environment
env_train = envs.PhasingEnv(config.panel_train,
                            min_graph_size=config.min_graph_size,
                            max_graph_size=config.max_graph_size,
                            skip_trivial_graphs=config.skip_trivial_graphs, graph_distribution=graph_dataset_indices)
agent = agents.DiscreteActorCriticAgent(env_train)
if config.pretrained_model is not None:
    agent.model.load_state_dict(torch.load(config.pretrained_model))

def compute_error_rates(solutions, validation_input_vcf):
    # given any subset of phasing solutions, computes errors rates against ground truth VCF
    idx2var = var.extract_variants(solutions)
    for v in idx2var.values():
        v.assign_haplotype()
    idx2var = utils.post_processing.update_split_block_phase_sets(agent.env.solutions, idx2var)
    vcf_writer.write_phased_vcf(validation_input_vcf, idx2var, config.validation_output_vcf)
    CHROM = benchmark.get_ref_name(config.validation_output_vcf)
    benchmark_result = benchmark.vcf_vcf_error_rate(config.validation_output_vcf, validation_input_vcf, indels=False)
    return CHROM, benchmark_result


def validate(model_checkpoint_id, episode_id):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    # TODO: pre-load the validation panel
    overall_sum_of_cuts = 0
    for validation_frag, validation_input_vcf in zip(open(config.panel_validation_frags, 'r').read().splitlines(), open(config.panel_validation_vcfs,'r').read().splitlines()):
        agent.env = envs.PhasingEnv(panel=validation_frag, skip_trivial_graphs=config.skip_trivial_graphs, skip_singleton_graphs=False, debug=True, record_solutions=True)
        sum_of_rewards = 0
        sum_of_cuts = 0
        episode_no = 0

        if config.debug:
            # ✨ W&B: Create a Table to store predictions for each test step
            columns=["id", "number of nodes", "number of edges", "density", "cut value"]
            test_table = wandb.Table(columns=columns)

            # solutions on graphs 0 to 10 nodes
            solutions_0_to_10 = []
            sum_of_cuts_0_to_10 = 0
            sum_of_rewards_0_to_10 = 0
            # solutions on graphs 11 to 50 nodes
            solutions_11_to_50 = []
            sum_of_cuts_11_to_50 = 0
            sum_of_rewards_11_to_50 = 0
            # solutions on graphs 51 to 100 nodes
            solutions_51_to_100 = []
            sum_of_cuts_51_to_100 = 0
            sum_of_rewards_51_to_100 = 0
            # solutions on graphs 101 to 200 nodes
            solutions_101_to_200 = []
            sum_of_cuts_101_to_200 = 0
            sum_of_rewards_101_to_200 = 0
            # solutions on graphs 201 to 500 nodes
            solutions_201_to_500 = []
            sum_of_cuts_201_to_500 = 0
            sum_of_rewards_201_to_500 = 0
            # solutions on graphs 501 to 1000 nodes
            solutions_501_to_1000 = []
            sum_of_cuts_501_to_1000 = 0
            sum_of_rewards_501_to_1000 = 0
            # solutions on graphs 1001 plus nodes
            solutions_1001_plus = []
            sum_of_cuts_1001_plus = 0
            sum_of_rewards_1001_plus = 0

        while agent.env.has_state():
            reward_val = agent.run_episode(config, test_mode=True)
            sum_of_rewards += reward_val
            cut_val = agent.env.get_cut_value()
            sum_of_cuts += cut_val
            if config.debug:
                graph_stats = agent.env.get_simple_graph_stats()
                #graph_stats = agent.env.get_graph_stats()
                test_table.add_data(episode_no, graph_stats["num_nodes"], graph_stats["num_edges"],
                                    graph_stats["density"], graph_stats["cut_value"])
                episode_num_nodes = agent.env.state.num_nodes
                episode_frags = agent.env.state.frag_graph.fragments
                if 0 <= episode_num_nodes <= 10:
                    solutions_0_to_10.append(episode_frags)
                    sum_of_cuts_0_to_10 += cut_val
                    sum_of_rewards_0_to_10 += reward_val
                elif 11 <= episode_num_nodes <= 50:
                    solutions_11_to_50.append(episode_frags)
                    sum_of_cuts_11_to_50 += cut_val
                    sum_of_rewards_11_to_50 += reward_val
                elif 51 <= episode_num_nodes <= 100:
                    solutions_51_to_100.append(episode_frags)
                    sum_of_cuts_51_to_100 += cut_val
                    sum_of_rewards_51_to_100 += reward_val
                elif 101 <= episode_num_nodes <= 200:
                    solutions_101_to_200.append(episode_frags)
                    sum_of_cuts_101_to_200 += cut_val
                    sum_of_rewards_101_to_200 += reward_val
                elif 201 <= episode_num_nodes <= 500:
                    solutions_201_to_500.append(episode_frags)
                    sum_of_cuts_201_to_500 += cut_val
                    sum_of_rewards_201_to_500 += reward_val
                elif 501 <= episode_num_nodes <= 1000:
                    solutions_501_to_1000.append(episode_frags)
                    sum_of_cuts_501_to_1000 += cut_val
                    sum_of_rewards_501_to_1000 += reward_val
                elif 1001 <= episode_num_nodes:
                    solutions_1001_plus.append(episode_frags)
                    sum_of_cuts_1001_plus += cut_val
                    sum_of_rewards_1001_plus += reward_val

            agent.env.reset()
            episode_no += 1

        overall_sum_of_cuts += sum_of_cuts

        def log_error_rates(solutions, sum_of_cuts, sum_of_rewards, descriptor="_default_", simple=False):
            CHROM, benchmark_result = compute_error_rates(solutions, validation_input_vcf)
            original_CHROM = CHROM
            switch_count = benchmark_result.switch_count[CHROM]
            mismatch_count = benchmark_result.mismatch_count[CHROM]
            flat_count = benchmark_result.flat_count[CHROM]
            phased_count = benchmark_result.phased_count[CHROM]
            AN50 = benchmark_result.get_AN50()
            N50 = benchmark_result.get_N50_phased_portion()
            CHROM = descriptor + CHROM

            with open(config.out_dir + "/benchmark.txt", "a") as out_file:
                out_file.write("benchmark of model: " + str(model_checkpoint_id) + descriptor + "\n")
                out_file.write("sum of cuts: " + str(sum_of_cuts) + "\n")
                out_file.write("switch count: " + str(switch_count) + "\n")
                out_file.write("mismatch count: " + str(mismatch_count) + "\n")
                out_file.write("flat count: " + str(flat_count) + "\n")
                out_file.write("phased count: " + str(phased_count) + "\n")
                out_file.write("AN50: " + str(AN50) + "\n")
                out_file.write("N50: " + str(N50) + "\n")
                out_file.write(str(benchmark_result) + "\n")

            if descriptor == "_default_":
                torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
                # log validation loop stats
                logging.getLogger(config_utils.MAIN_LOG).info("Validation checkpoint: %d, Sum of Cuts: %d, Sum of Rewards: %d, Switch Count: %d,"
                                                              " Mismatch Count: %d, Flat Count: %d, Phased Count: %d, AN50: %d, N50: %d" %
                                                              (model_checkpoint_id, sum_of_cuts, sum_of_rewards, switch_count, mismatch_count, flat_count, phased_count, AN50, N50))

            logging.getLogger(config_utils.STATS_LOG_VALIDATE).info("%s,%s,%s,%s, %s,%s,%s,%s,%s,%s" % (CHROM, episode_id, sum_of_cuts, sum_of_rewards, switch_count, mismatch_count, flat_count, phased_count, AN50, N50))

            wandb.log({"Episode": episode_id,"Validation Sum of Rewards on " + CHROM: sum_of_rewards})
            wandb.log({"Episode": episode_id,"Validation Sum of Cuts on " + CHROM: sum_of_cuts})
            wandb.log({"Episode": episode_id, "Validation Switch Count on " + CHROM: switch_count})
            wandb.log({"Episode": episode_id, "Validation Mismatch Count on " + CHROM: mismatch_count})
            wandb.log({"Episode": episode_id, "Validation Flat Count on " + CHROM: flat_count})
            wandb.log({"Episode": episode_id, "Validation Phased Count on " + CHROM: phased_count})
            wandb.log({"Episode": episode_id, "Validation AN50 on " + CHROM: AN50})
            wandb.log({"Episode": episode_id, "Validation AN50 on " + CHROM: N50})

            if descriptor == "_default_":
                wandb.log({"Episode": episode_id,"validation_predictions_" + CHROM + "_" + str(model_checkpoint_id): test_table})
            # output the phased VCF (phase blocks)
            return original_CHROM

        original_CHROM = log_error_rates(agent.env.solutions, sum_of_cuts, sum_of_rewards, "_default_")
        wandb.log({"Episode": episode_id, "Validation Sum of Rewards on " + "_0to10Nodes_"+ original_CHROM: sum_of_rewards_0_to_10})
        wandb.log({"Episode": episode_id, "Validation Sum of Cuts on " + "_0to10Nodes_"+ original_CHROM: sum_of_cuts_0_to_10})
        wandb.log({"Episode": episode_id, "Validation Sum of Rewards on " + "_11_to_50_" + original_CHROM: sum_of_rewards_11_to_50})
        wandb.log({"Episode": episode_id, "Validation Sum of Cuts on " + "_11_to_50_" + original_CHROM: sum_of_cuts_11_to_50})
        wandb.log({"Episode": episode_id, "Validation Sum of Rewards on " + "_51_to_100_" + original_CHROM: sum_of_rewards_51_to_100})
        wandb.log({"Episode": episode_id, "Validation Sum of Cuts on " + "_51_to_100_" + original_CHROM: sum_of_cuts_51_to_100})
        wandb.log({"Episode": episode_id, "Validation Sum of Rewards on " + "_101_to_200_" + original_CHROM: sum_of_rewards_101_to_200})
        wandb.log({"Episode": episode_id, "Validation Sum of Cuts on " + "_101_to_200_" + original_CHROM: sum_of_cuts_101_to_200})
        wandb.log({"Episode": episode_id, "Validation Sum of Rewards on " + "_201_to_500_" + original_CHROM: sum_of_rewards_201_to_500})
        wandb.log({"Episode": episode_id, "Validation Sum of Cuts on " + "_201_to_500_" + original_CHROM: sum_of_cuts_201_to_500})
        wandb.log({"Episode": episode_id, "Validation Sum of Rewards on " + "_501_to_1000_" + original_CHROM: sum_of_rewards_501_to_1000})
        wandb.log({"Episode": episode_id, "Validation Sum of Cuts on " + "_501_to_1000_" + original_CHROM: sum_of_cuts_501_to_1000})
        wandb.log({"Episode": episode_id, "Validation Sum of Rewards on " + "_1001_plus_" + original_CHROM: sum_of_rewards_1001_plus})
        wandb.log({"Episode": episode_id, "Validation Sum of Cuts on " + "_1001_plus_" + original_CHROM: sum_of_cuts_1001_plus})

        """"
        if config.debug:
            log_error_rates(solutions_0_to_10, sum_of_cuts_0_to_10, sum_of_rewards_0_to_10, "_0to10Nodes_")
            log_error_rates(solutions_11_to_50, sum_of_cuts_11_to_50, sum_of_rewards_11_to_50, "_11to50Nodes_")
            log_error_rates(solutions_51_to_100, sum_of_cuts_51_to_100, sum_of_rewards_51_to_100, "_51to100Nodes_")
            log_error_rates(solutions_101_to_200, sum_of_cuts_101_to_200, sum_of_rewards_101_to_200, "_101to200Nodes_")
            log_error_rates(solutions_201_to_500, sum_of_cuts_201_to_500, sum_of_rewards_201_to_500, "_201to500Nodes_")
            log_error_rates(solutions_501_to_1000, sum_of_cuts_501_to_1000, sum_of_rewards_501_to_1000, "_501to1000Nodes_")
            log_error_rates(solutions_1001_plus, sum_of_cuts_1001_plus, sum_of_rewards_1001_plus, "_1001_plus_")
        """

    return overall_sum_of_cuts

# Run the training
best_validation_reward = 0
model_checkpoint_id = 0
episode_id = 0
while agent.env.has_state():
    if config.max_episodes is not None and episode_id >= config.max_episodes:
        break
    episode_reward = agent.run_episode(config, episode_id=episode_id)
    if episode_id % config.interval_validate == 0 and config.panel_validation_frags is not None:
        reward = validate(model_checkpoint_id, episode_id)
        model_checkpoint_id += 1
        if reward > best_validation_reward:
            best_validation_reward = reward
            torch.save(agent.model.state_dict(), config.best_model_path)
    episode_id += 1
    agent.env = env_train
    agent.env.reset()

# save the model
torch.save(agent.model.state_dict(), config.model_path)
