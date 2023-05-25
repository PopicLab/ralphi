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
import models.actor_critic as agents
import torch
from joblib import Parallel, delayed


def compute_error_rates(solutions, validation_input_vcf, agent, config, genome, group):
    # given any subset of phasing solutions, computes errors rates against ground truth VCF
    idx2var = var.extract_variants(solutions)
    for v in idx2var.values():
        v.assign_haplotype()
    idx2var = utils.post_processing.update_split_block_phase_sets(agent.env.solutions, idx2var)
    output_vcf = config.validation_output_vcf + "_" +  str(group) + ".vcf"
    vcf_writer.write_phased_vcf(validation_input_vcf, idx2var, output_vcf)
    chrom = benchmark.get_ref_name(output_vcf)
    benchmark_result = benchmark.vcf_vcf_error_rate(output_vcf, validation_input_vcf, indels=False)
    if not config.light_logging:
        hap_blocks = hap_block_visualizer.pretty_print(solutions, idx2var.items(), validation_input_vcf, genome)
        return chrom, benchmark_result, hap_blocks
    else:
        return chrom, benchmark_result, None

def log_error_rates(solutions, input_vcf, sum_of_cuts, sum_of_rewards, model_checkpoint_id, episode_id, agent, config, genome, group, descriptor="_default_"):
    chrom, benchmark_result, hap_blocks = compute_error_rates(solutions, input_vcf, agent, config, genome, group)
    if not config.light_logging:
        AN50 = benchmark_result.get_AN50()
        N50 = benchmark_result.get_N50_phased_portion()
        label = descriptor + ", " + chrom

        with open(config.out_dir + "/benchmark_" + str(group) + ".txt", "a") as out_file:
            out_file.write("benchmark of model: " + str(model_checkpoint_id) + "\n")
            out_file.write("switch count: " + str(benchmark_result.switch_count[chrom]) + "\n")
            out_file.write("mismatch count: " + str(benchmark_result.mismatch_count[chrom]) + "\n")
            out_file.write("switch loc: " + str(benchmark_result.switch_loc[chrom]) + "\n")
            out_file.write("mismatch loc: " + str(benchmark_result.mismatch_loc[chrom]) + "\n")
            out_file.write("flat count: " + str(benchmark_result.flat_count[chrom]) + "\n")
            out_file.write("phased count: " + str(benchmark_result.phased_count[chrom]) + "\n")
            out_file.write("AN50: " + str(AN50) + "\n")
            out_file.write("N50: " + str(N50) + "\n")
            out_file.write("sum of cuts: " + str(sum_of_cuts) + "\n")
            out_file.write(descriptor + "\n")
            out_file.write(str(benchmark_result) + "\n")
            if benchmark_result.switch_count[chrom] > 0 or benchmark_result.mismatch_count[chrom] > 0:
                out_file.write(str(hap_blocks) + "\n")

        logging.getLogger(config_utils.STATS_LOG_VALIDATE).info("%s,%s,%s,%s, %s,%s,%s,%s,%s,%s"
                                                                % (label, episode_id, sum_of_cuts, sum_of_rewards,
                                                                   benchmark_result.switch_count[chrom],
                                                                   benchmark_result.mismatch_count[chrom],
                                                                   benchmark_result.flat_count[chrom],
                                                                   benchmark_result.phased_count[chrom], AN50, N50))
    # output the phased VCF (phase blocks)
    return chrom, benchmark_result.switch_count[chrom], benchmark_result.mismatch_count[chrom], benchmark_result.flat_count[chrom], benchmark_result.phased_count[chrom]


def validation_task(model_checkpoint_id, episode_id, sub_df, model_path, config, group):
    task_component_stats = []
    agent = None
    for index, component_row in tqdm.tqdm(sub_df.iterrows()):
        with open(component_row.component_path, 'rb') as f:
            subgraph = pickle.load(f)
            subgraph.indexed_graph_stats = component_row

            mini_env = envs.PhasingEnv(config, preloaded_graphs=subgraph, record_solutions=True)
            if agent is not None:
                agent.env = mini_env
            else:
                agent = agents.DiscreteActorCriticAgent(mini_env)
                agent.model.load_state_dict(torch.load(model_path))

            sum_of_rewards = 0
            sum_of_cuts = 0
            reward_val = agent.run_episode(config, test_mode=True)
            sum_of_rewards += reward_val
            cut_val = agent.env.get_cut_value()
            sum_of_cuts += cut_val
            if config.debug:
                graph_stats = agent.env.state.frag_graph.graph_properties

                graph_path = os.path.split(component_row.component_path)[1] + str(graph_stats)
                graph_stats["cut_value"] = cut_val

                if not config.light_logging:
                    wandb.log({"Episode": episode_id,
                               "Cut Value on: " + str(component_row.genome) + "_" + str(component_row.coverage) + "_" + str(
                                   component_row.error_rate) + "_" + graph_path: graph_stats["cut_value"]})
                vcf_path = component_row.component_path + ".vcf"

                ch, sw, mis, flat, phased = log_error_rates([agent.env.state.frag_graph.fragments], vcf_path,
                                                            cut_val, reward_val, model_checkpoint_id, episode_id, agent,
                                                            config, component_row.genome, group, graph_path)

                cur_index = component_row.values.tolist()
                cur_index.extend([sw, mis, flat, phased, reward_val, cut_val, ch])

                task_component_stats.append(cur_index)
    return pd.DataFrame(task_component_stats, columns=list(sub_df.columns) + ["switch", "mismatch", "flat", "phased", "reward_val", "cut_val", "chr"])
def validate(model_checkpoint_id, episode_id, validation_dataset, config):
    # benchmark the current model against a held out set of fragment graphs (validation panel)

    print("running validation with model number:  ", model_checkpoint_id, ", at episode: ", episode_id)

    input_tuples = []
    for i, sub_df in enumerate(validation_dataset):
        input_tuples.append((model_checkpoint_id, episode_id, sub_df, "%s/dphase_model_%d.pt" % (config.out_dir, model_checkpoint_id), config, str(i)))
    
    validation_component_stats = Parallel(n_jobs=config.num_cores_validation)(delayed(validation_task)(input_elem) for input_elem in input_tuples)

    validation_indexing_df = pd.concat(validation_component_stats)
    validation_indexing_df.to_pickle("%s/validation_index_for_model_%d.pickle" % (config.out_dir, model_checkpoint_id))

    def log_stats_for_filter(validation_filtered_df, descriptor="Overall"):
        metrics_of_interest = ["reward_val", "cut_val", "switch", "mismatch", "flat", "phased"]
        for metric in metrics_of_interest:
            wandb.log({"Episode": episode_id, descriptor + " Validation " + metric + " on " + "_default_overall": validation_filtered_df[metric].sum()})
        wandb.log({"Episode": episode_id, descriptor + " Validation " + "Number of Examples" + " on " + "_default_overall": len(validation_filtered_df)})
        percentage_metrics = ["switch", "mismatch", "flat"]
        for metric in percentage_metrics:
            wandb.log({"Episode": episode_id, descriptor + " Validation " + "% with " + metric + " errors" + " on " + "_default_overall": len(validation_filtered_df[validation_filtered_df[metric] > 0]) / len(validation_filtered_df)})

    # stats for entire validation set
    log_stats_for_filter(validation_indexing_df) 
    
    # log stats for graphs from each quantile of each graph property specified in the validation config
    keys = validation_indexing_df.group.unique()
    for group in validation_indexing_df.group.unique():
        log_stats_for_filter(validation_indexing_df[validation_indexing_df["group"] == group], "group: " + str(group))

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
