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
import pickle
import tqdm

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

# set up performance tracking
if config.debug:
    wandb.init(project="dphase_experiments", entity="dphase", dir=config.log_dir)
else:
    # automatically results in ignoring all wandb calls
    wandb.init(project="dphase_experiments", entity="dphase", dir=config.log_dir, mode="disabled")


training_dataset = dataset_gen.graph_generator.GraphDistribution(config.panel_train, load_components=True, store_components=True, save_indexes=True, compress=config.compress)

validation_dataset = dataset_gen.graph_generator.GraphDistribution(fragment_files_panel=config.panel_validation_frags, vcf_panel=config.panel_validation_vcfs,
                                                                            load_components=True,
                                                                            store_components=True,
                                                                            save_indexes=True, compress=config.compress)
# e.g. to only validate on cases with articulation points
# validation_dataset = validation_dataset[validation_dataset["articulation_points"] != 0]


# Setup the agent and the training environment
env_train = envs.PhasingEnv(config.panel_train,
                            min_graph_size=config.min_graph_size,
                            max_graph_size=config.max_graph_size,
                            skip_trivial_graphs=config.skip_trivial_graphs, graph_distribution=training_dataset, compress=config.compress)
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
        out_file.write("benchmark of model: " + str(model_checkpoint_id) + "\n")
        out_file.write("switch count: " + str(switch_count) + "\n")
        out_file.write("mismatch count: " + str(mismatch_count) + "\n")
        out_file.write("switch loc: " + str(switch_loc) + "\n")
        out_file.write("mismatch loc: " + str(mismatch_loc) + "\n")
        out_file.write("flat count: " + str(flat_count) + "\n")
        out_file.write("phased count: " + str(phased_count) + "\n")
        out_file.write("AN50: " + str(AN50) + "\n")
        out_file.write("N50: " + str(N50) + "\n")
        out_file.write("sum of cuts: " + str(sum_of_cuts) + "\n")
        out_file.write(descriptor + "\n")
        out_file.write(str(benchmark_result) + "\n")
        if switch_count > 0 or mismatch_count > 0:
            out_file.write(str(hap_blocks) + "\n")

    logging.getLogger(config_utils.STATS_LOG_VALIDATE).info("%s,%s,%s,%s, %s,%s,%s,%s,%s,%s" % (
    CHROM, episode_id, sum_of_cuts, sum_of_rewards, switch_count, mismatch_count, flat_count, phased_count, AN50, N50))
    # output the phased VCF (phase blocks)
    return original_CHROM, switch_count, mismatch_count, flat_count, phased_count

def validate(model_checkpoint_id, episode_id):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    validation_component_stats = []
    for index, component_row in tqdm.tqdm(validation_dataset.iterrows()):
        with open(component_row.component_path + ".vcf.graph", 'rb') as f:
            subgraph = pickle.load(f)
            subgraph.indexed_graph_stats = component_row
            if subgraph.n_nodes < 2 and subgraph.fragments[0].n_variants < 2:
                # only validate on non-singleton graphs with > 1 variant
                continue
            print("validating on subgraph with ", subgraph.n_nodes, " nodes...")
            mini_env = envs.PhasingEnv(preloaded_graphs=[subgraph], record_solutions=True)
            agent.env = mini_env
            sum_of_rewards = 0
            sum_of_cuts = 0
            reward_val = agent.run_episode(config, test_mode=True)
            sum_of_rewards += reward_val
            cut_val = agent.env.get_cut_value()
            sum_of_cuts += cut_val
            if config.debug:
                graph_stats = agent.env.get_indexed_graph_stats()
                graph_path = os.path.split(component_row.component_path)[1] + "_Nodes_" + str(graph_stats["n_nodes"]) + "_Edges_" + str(
                    graph_stats["n_edges"]) + "_Density_" + str(graph_stats["density"]) + "_Articulation_" + str(graph_stats["articulation_points"]) \
                             + "_Diameter_" + str(graph_stats["diameter"]) + "_NodeConnectivity_" + str(graph_stats["node connectivity"]) \
                             + "_EdgeConnectivity_" + str(graph_stats["edge_connectivity"])

                wandb.log({"Episode": episode_id,
                           "Cut Value on: " + graph_path: graph_stats["cut_value"]})
                vcf_path = component_row.component_path + ".vcf"

                ch, sw, mis, flat, phased = log_error_rates([agent.env.state.frag_graph.fragments], vcf_path,
                                                            cut_val, reward_val, graph_path)

                cur_index = component_row.values.tolist()
                cur_index.append(sw)
                cur_index.append(mis)
                cur_index.append(flat)
                cur_index.append(phased)
                cur_index.append(reward_val)
                cur_index.append(cut_val)
                cur_index.append(ch)
                validation_component_stats.append(cur_index)

    validation_indexing_df = pd.DataFrame(validation_component_stats,
                                   columns=["component_path", "index", "n_nodes", "n_edges", "density",
                                            "articulation_points", "node connectivity", "edge_connectivity", "diameter",
                                            "min_degree", "max_degree", "pos_edges", "neg_edges",
                                            "sum_of_pos_edge_weights", "sum_of_neg_edge_weights",
                                            "trivial", "switch", "mismatch", "flat", "phased", "reward_val", "cut_val", "chr"])
    validation_indexing_df.to_pickle("%s/validation_index_for_model_%d.pickle" % (config.out_dir, model_checkpoint_id))

    def log_stats_for_filter(validation_filtered_df, descriptor="Pandas"):
        wandb.log({"Episode": episode_id, descriptor + " Validation Sum of Rewards on " + "_default_overall": validation_filtered_df["reward_val"].sum()})
        wandb.log({"Episode": episode_id, descriptor + " Validation Sum of Cuts on " + "_default_overall": validation_filtered_df["cut_val"].sum()})
        wandb.log({"Episode": episode_id, descriptor + " Validation Switch Count on " + "_default_overall": validation_filtered_df["switch"].sum()})
        wandb.log({"Episode": episode_id, descriptor + " Validation Mismatch Count on " + "_default_overall": validation_filtered_df["mismatch"].sum()})
        wandb.log({"Episode": episode_id, descriptor + " Validation Flat Count on " + "_default_overall": validation_filtered_df["flat"].sum()})
        wandb.log({"Episode": episode_id, descriptor + " Validation Phased Count on " + "_default_overall": validation_filtered_df["phased"].sum()})

    # stats for entire validation set
    log_stats_for_filter(validation_indexing_df, "Overall")

    # log specific plots to wandb for graph topologies we are interested in
    articulation_df = validation_indexing_df.loc[validation_indexing_df["articulation_points"] > 0]
    log_stats_for_filter(articulation_df, "Articulation > 0:")
    articulation_df = validation_indexing_df.loc[validation_indexing_df["articulation_points"] == 0]
    log_stats_for_filter(articulation_df, "Articulation == 0:")
    diameter_df = validation_indexing_df.loc[validation_indexing_df["diameter"] <= 5]
    log_stats_for_filter(diameter_df, "Diameter <= 5:")
    node_filter_df = validation_indexing_df.loc[(0 <= validation_indexing_df["n_nodes"])
                                                      & (validation_indexing_df["n_nodes"] <= 100)]
    log_stats_for_filter(node_filter_df, "0 to 100 nodes:")
    node_filter_df = validation_indexing_df.loc[(101 <= validation_indexing_df["n_nodes"])]
    log_stats_for_filter(node_filter_df, "101 plus nodes:")

    torch.save(agent.model.state_dict(), "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id))
    return validation_indexing_df["reward_val"].sum()


# Run the training
best_validation_reward = 0
model_checkpoint_id = 0
episode_id = 0
while agent.env.has_state():
    if config.max_episodes is not None and episode_id >= config.max_episodes:
        break
    episode_reward = agent.run_episode(config, episode_id=episode_id)
    if episode_id % config.interval_validate == 0:
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
