import third_party.HapCUT2.utilities.calculate_haplotype_statistics as benchmark
import utils.hap_block_visualizer as hap_block_visualizer
import pickle
import utils.post_processing
import seq.phased_vcf as vcf_writer
import pandas as pd
import engine.config as config_utils
import wandb
import logging
import seq.var as var
import tqdm
import envs.phasing_env as envs
import os

def compute_error_rates(solutions, validation_input_vcf, agent, config):
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

def log_error_rates(solutions, input_vcf, sum_of_cuts, sum_of_rewards, model_checkpoint_id, episode_id, agent, config, descriptor="_default_"):
    CHROM, benchmark_result, hap_blocks = compute_error_rates(solutions, input_vcf, agent, config)
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

def validate(model_checkpoint_id, episode_id, validation_dataset, agent, config):
    # benchmark the current model against a held out set of fragment graphs (validation panel)
    validation_component_stats = []
    print("running validation with model number:  ", model_checkpoint_id, ", at episode: ", episode_id)
    for index, component_row in tqdm.tqdm(validation_dataset.iterrows()):
        with open(component_row.component_path + ".vcf.graph", 'rb') as f:
            subgraph = pickle.load(f)
            subgraph.indexed_graph_stats = component_row
            if subgraph.n_nodes < 2 and subgraph.fragments[0].n_variants < 2:
                # only validate on non-singleton graphs with > 1 variant
                continue
            mini_env = envs.PhasingEnv(preloaded_graphs=[subgraph], record_solutions=True)
            agent.env = mini_env
            sum_of_rewards = 0
            sum_of_cuts = 0
            reward_val = agent.run_episode(config, test_mode=True)
            sum_of_rewards += reward_val
            cut_val = agent.env.get_cut_value()
            sum_of_cuts += cut_val
            if config.debug:
                graph_stats = agent.env.get_graph_stats().query_stats()

                graph_path = os.path.split(component_row.component_path)[1] + str(graph_stats)
                graph_stats["cut_value"] = agent.env.get_cut_value()
                wandb.log({"Episode": episode_id,
                           "Cut Value on: " + graph_path: graph_stats["cut_value"]})
                vcf_path = component_row.component_path + ".vcf"

                ch, sw, mis, flat, phased = log_error_rates([agent.env.state.frag_graph.fragments], vcf_path,
                                                            cut_val, reward_val, model_checkpoint_id, episode_id, agent, config, graph_path)

                cur_index = component_row.values.tolist()
                cur_index.extend([sw, mis, flat, phased, reward_val, cut_val, ch])
                validation_component_stats.append(cur_index)

    print("columns:", list(validation_dataset.columns))
    print("validation component stats: ", validation_component_stats)
    validation_indexing_df = pd.DataFrame(validation_component_stats,
                                   columns=list(validation_dataset.columns) + ["switch", "mismatch", "flat", "phased", "reward_val", "cut_val", "chr"])
    validation_indexing_df.to_pickle("%s/validation_index_for_model_%d.pickle" % (config.out_dir, model_checkpoint_id))

    def log_stats_for_filter(validation_filtered_df, descriptor="Pandas"):
        metrics_of_interest = ["reward_val", "cut_val", "switch", "mismatch", "flat", "phased"]
        for metric in metrics_of_interest:
            wandb.log({"Episode": episode_id, descriptor + " Validation " + metric + " on " + "_default_overall": validation_filtered_df[metric].sum()})

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

    return validation_indexing_df["reward_val"].sum()