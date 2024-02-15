import argparse
import torch
import time
import models.actor_critic as algs
import envs.phasing_env as envs
import seq.var as var
import seq.phased_vcf as vcf_writer
import utils.post_processing
import pickle
import engine.config as config_utils
import logging

# ------ CLI ------
parser = argparse.ArgumentParser(description='Test haplotype phasing')
parser.add_argument('--config', help='Testing configuration YAML')
args = parser.parse_args()
# -----------------
total_start_time = time.time()

config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.TEST)

torch.set_num_threads(config.num_cores_torch)
env = envs.PhasingEnv(config, record_solutions=True)
agent = algs.DiscreteActorCriticAgent(env)
agent.model.load_state_dict(torch.load(config.model))

# run through all the components of the fragment graph
n_episodes = 0
while env.has_state():
    if env.state.frag_graph.trivial:
        logging.debug("Component is trivial")
        env.lookup_error_free_instance()
        n_episodes += 1
        env.reset()
        continue
    logging.debug("Component is non-trivial")
    start_time = time.time()
    done = False
    reward = agent.run_episode(config, test_mode=True)
    logging.debug("Runtime: %f " % (time.time() - start_time))
    env.reset()
    n_episodes += 1
logging.info("Total number of episodes: %d " % n_episodes)

with open(config.phasing_output_path, 'wb') as phased_output:
    pickle.dump(env.solutions, phased_output, protocol=pickle.HIGHEST_PROTOCOL)

if config.articulation_stitch:
    var.stitch_fragments(env.solutions)

# output the phased VCF (phase blocks)
idx2var = var.extract_variants(env.solutions)
for v in idx2var.values():
    v.assign_haplotype()

if config.postprocess_ambiguous:
    idx2var = utils.post_processing.postprocess_ambiguous(env.solutions, idx2var, config.evidence_threshold)
    logging.info("Post-processed blocks that were split up due to ambiguous variants")

logging.info("Overall Runtime: %f " % (time.time() - total_start_time))
vcf_writer.write_phased_vcf(config.input_vcf, idx2var, config.output_vcf)
