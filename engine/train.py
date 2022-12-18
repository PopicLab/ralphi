import argparse
import pandas as pd
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
import utils.hap_block_visualizer as hap_block_visualizer

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
    wandb.init(project="chr1_every_graph_leq_100", entity="dphase", dir=config.log_dir)
else:
    # automatically results in ignoring all wandb calls
    wandb.init(project="chr1_every_graph_leq_100", entity="dphase", dir=config.log_dir, mode="disabled")


graph_dataset_indices = None
if config.define_training_distribution:
    training_distribution = dataset_gen.graph_generator.GraphDistribution(config.panel_train, load_components=True, store_components=True, save_indexes=True)
    graph_dataset_indices = training_distribution.load_graph_dataset_indices()

"""
graph_size_lower_bound = config.min_graph_size #int(graph_dataset_indices["n_nodes"].min())
graph_size_upper_bound = config.max_graph_size #int(graph_dataset_indices["n_nodes"].max())
bucket_size = int((graph_size_upper_bound - graph_size_lower_bound) / 10)

curriculum_learning_indices = []
for i in range(graph_size_lower_bound, graph_size_upper_bound, bucket_size):
    graphs_within_range = graph_dataset_indices[
        (graph_dataset_indices.n_nodes > i) & (graph_dataset_indices.n_nodes < i + bucket_size)]
    if not graphs_within_range.empty:
        curriculum_learning_indices.append(graphs_within_range.sample(n=10000, replace=True, random_state=1))
curriculum_learning_indices = pd.concat(curriculum_learning_indices)
"""

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
    hap_blocks = hap_block_visualizer.pretty_print(solutions, idx2var.items(), validation_input_vcf)
    return CHROM, benchmark_result, hap_blocks


def validate(model_checkpoint_id, episode_id):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    # TODO: pre-load the validation panel
    overall_sum_of_cuts = 0
    iteration = 0
    for validation_frag, validation_input_vcf in zip(open(config.panel_validation_frags, 'r').read().splitlines(), open(config.panel_validation_vcfs,'r').read().splitlines()):
        validation_dataset_indices = None
        graph_dataset = None
        if config.define_validation_distribution:
            #TODO: optimize to only load in these validation graphs once to save on I/O
            validation_distribution = dataset_gen.graph_generator.GraphDistribution(validation_frag,
                                                                                     load_components=True,
                                                                                     store_components=True,
                                                                                     save_indexes=True)
            validation_dataset_indices = validation_distribution.load_graph_dataset_indices()
            graph_dataset=validation_dataset_indices[(validation_dataset_indices.n_nodes <= 100)]
            """
	    graph_size_lower_bound = int(validation_dataset_indices["n_nodes"].min())
            graph_size_upper_bound = int(validation_dataset_indices["n_nodes"].max())
            bucket_size = int((graph_size_upper_bound - graph_size_lower_bound) / 10)
            graph_dataset = []

            for i in range(graph_size_lower_bound, graph_size_upper_bound, bucket_size):
                graphs_within_range = validation_dataset_indices[(validation_dataset_indices.n_nodes > i) & (validation_dataset_indices.n_nodes < i + bucket_size)]
                if not graphs_within_range.empty:
                    graph_dataset.append(graphs_within_range.sample(n=1, replace=True, random_state=1))

            for i in range(0, 100, 10):
                graphs_within_range = validation_dataset_indices[
                    (validation_dataset_indices.density > (i/100)) & (validation_dataset_indices.density < ((i + 10)/100))]
                if not graphs_within_range.empty:
                    graph_dataset.append(graphs_within_range.sample(n=1, replace=True, random_state=1))

	
            graph_dataset = pd.concat(graph_dataset)
            """
        agent.env = envs.PhasingEnv(panel=validation_frag, skip_trivial_graphs=config.skip_trivial_graphs, skip_singleton_graphs=False, debug=True, record_solutions=True, graph_distribution=graph_dataset)
        sum_of_rewards = 0
        sum_of_cuts = 0
        episode_no = 0

        if config.debug:
            # ✨ W&B: Create a Table to store predictions for each test step
            columns=["id", "number of nodes", "number of edges", "density", "cut value", "raw cut", "switch_count", "mismatch_count", "flat_count", "phased_count"]
            test_table = wandb.Table(columns=columns)

        def log_error_rates(solutions, input_vcf, sum_of_cuts, sum_of_rewards, descriptor="_default_", simple=False):
            CHROM, benchmark_result, hap_blocks = compute_error_rates(solutions, input_vcf)
            original_CHROM = CHROM
            switch_count = benchmark_result.switch_count[CHROM]
            switch_loc = benchmark_result.switch_loc[CHROM]
            mismatch_count = benchmark_result.mismatch_count[CHROM]
            mismatch_loc = benchmark_result.mismatch_loc[CHROM]
            flat_count = benchmark_result.flat_count[CHROM]
            phased_count = benchmark_result.phased_count[CHROM]
            AN50 = benchmark_result.get_AN50()
            N50 = benchmark_result.get_N50_phased_portion()
            CHROM = descriptor + ", " + CHROM

            with open(config.out_dir + "/benchmark.txt", "a") as out_file:
                out_file.write("benchmark of model: " + str(model_checkpoint_id) + descriptor + "\n")
                out_file.write("sum of cuts: " + str(sum_of_cuts) + "\n")
                out_file.write("switch count: " + str(switch_count) + "\n")
                out_file.write("mismatch count: " + str(mismatch_count) + "\n")
                out_file.write("switch loc: " + str(switch_loc) + "\n")
                out_file.write("mismatch loc: " + str(mismatch_loc) + "\n")
                out_file.write("flat count: " + str(flat_count) + "\n")
                out_file.write("phased count: " + str(phased_count) + "\n")
                out_file.write("AN50: " + str(AN50) + "\n")
                out_file.write("N50: " + str(N50) + "\n")
                out_file.write(str(benchmark_result) + "\n")
                out_file.write(str(hap_blocks) + "\n")

            if descriptor == "_default_":
                #torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
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
            #test_table.add_data(episode_no, graph_stats["num_nodes"], graph_stats["num_edges"],
            #                    graph_stats["density"], graph_stats["cut_value"],
            #                    agent.env.state.g.ndata['x'][:, 0].cpu().numpy().tolist(), switch_count, mismatch_count, flat_count, phased_count)
            if descriptor == "_default_":
                wandb.log({"Episode": episode_id,"validation_predictions_" + CHROM + "_" + str(model_checkpoint_id): test_table})
            # output the phased VCF (phase blocks)
            return original_CHROM, switch_count, mismatch_count, flat_count, phased_count

        total_sum_of_cuts = 0
        total_sum_of_rewards = 0
        total_switch = 0
        total_mismatch = 0
        total_flat = 0
        total_phased = 0
        while agent.env.has_state():
            reward_val = agent.run_episode(config, test_mode=True)
            sum_of_rewards += reward_val
            cut_val = agent.env.get_cut_value()
            sum_of_cuts += cut_val
            if config.debug:
                graph_stats = agent.env.get_graph_stats()
                graph_path = "_Nodes_" + str(graph_stats["num_nodes"]) + "_Edges_" + str(
                    graph_stats["num_edges"]) + "_Density_" + str(graph_stats["density"])

                wandb.log({"Episode": episode_id,
                           "Cut Value on: " + graph_path: graph_stats["cut_value"]})

                #wandb.log({"Episode": episode_id,
                #           "Raw Cut on: " + graph_path: agent.env.state.g.ndata['x'][:, 0].cpu().numpy().tolist()})

                vcf_path = config.out_dir + str(episode_no) + graph_path + ".vcf"
                agent.env.state.frag_graph.construct_vcf_for_specific_frag_graph(validation_input_vcf,
                                                                                    vcf_path)
                
                ch, sw, mis, flat, phased = log_error_rates([agent.env.state.frag_graph.fragments], vcf_path, cut_val, reward_val, graph_path)
                total_sum_of_cuts += cut_val
                total_sum_of_rewards = reward_val
                total_switch += sw
                total_mismatch += mis
                total_flat += flat
                total_phased += phased

            agent.env.reset()
            episode_no += 1

        overall_sum_of_cuts += sum_of_cuts
        #wandb.log({"Episode": episode_id,
        #           "validation_predictions_" + str(model_checkpoint_id): test_table})
        if iteration == 0:
            wandb.log({"Episode": episode_id,"Validation Sum of Rewards on " + "_default_chr1": total_sum_of_rewards})
            wandb.log({"Episode": episode_id,"Validation Sum of Cuts on " + "_default_chr1": total_sum_of_cuts})
            wandb.log({"Episode": episode_id, "Validation Switch Count on " + "_default_chr1": total_switch})
            wandb.log({"Episode": episode_id, "Validation Mismatch Count on " + "_default_chr1": total_mismatch})
            wandb.log({"Episode": episode_id, "Validation Flat Count on " + "_default_chr1": total_flat})
            wandb.log({"Episode": episode_id, "Validation Phased Count on " + "_default_chr1": total_phased}) 
            #wandb.log({"Episode": episode_id, "validation_predictions_" + "chr1_"  + str(model_checkpoint_id): test_table})
        else:
            wandb.log({"Episode": episode_id, "validation_predictions_" + "chr6_"  + str(model_checkpoint_id): test_table})
        iteration += 1        
        #original_CHROM = log_error_rates(agent.env.solutions, validation_input_vcf, sum_of_cuts, sum_of_rewards, "_default_")
    torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
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
