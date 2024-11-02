import argparse
import torch
import engine
from copy import deepcopy
import models.actor_critic
import envs.phasing_env
import seq.var as var
import engine.config as config_utils
import seq.io as vcf_writer
from seq.frags import generate_fragments
import logging
import numpy as np
from joblib import Parallel, delayed
import pickle
import tqdm

print("***********************************************")
print("*  ralphi (%s): haplotype assembly mode *" % engine.__version__)
print("***********************************************")
parser = argparse.ArgumentParser(description='Haplotype assembly')
parser.add_argument('--config', help='YAML config file')
args = parser.parse_args()


def phase(chr_names, config):  # runs phasing on the specified list of chromosomes
    logging.root.setLevel(logging.getLevelName(config.logging_level))
    for chromosome in chr_names:
        # -------- load variants and reads to generate input fragments
        config.fragments = generate_fragments(config, chromosome)
        # -------- initialize the fragment graph environment and load the pre-trained agent
        env = envs.phasing_env.PhasingEnv(config, record_solutions=True)
        agent = models.actor_critic.DiscreteActorCriticAgent(env)
        agent.model.load_state_dict(torch.load(config.model))
        # -------- run ralphi on all the connected components of the fragment graph
        while env.has_state():
            if env.state.frag_graph.trivial: env.process_error_free_instance()
            else: agent.run_episode(config, test_mode=True)
            env.reset()
        # ------- output the phased VCF
        logging.info("Finished phasing, writing outputs for %s" % chromosome)
        idx2var = var.extract_variants(env.solutions)
        for v in idx2var.values():
            v.assign_haplotype()
        with open("%s/%s.pkl" % (config.out_dir, chromosome), 'wb') as chr_out:
             pickle.dump(idx2var, chr_out)
        vcf_writer.write_phased_vcf(config.vcf, idx2var,
                                    "%s/%s.ralphi.vcf" % (config.out_dir, chromosome), chromosome)
        logging.info("Finished processing %s" % chromosome)


config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.TEST)
torch.set_num_threads(config.num_cores_torch)
logging.info("Running on %d processes" % config.n_procs)
chr_name_chunks = np.array_split(np.array(config.chr_names), config.n_procs)
logging.info("Chromosomes/process partition: " + str([np.array2string(chk) for chk in chr_name_chunks]))
Parallel(n_jobs=config.n_procs)(delayed(phase)(chr_name_chunks[i], deepcopy(config)) for i in range(config.n_procs))

# logging.info("Merging results")
# phasing_result = {}
# for chromosome in tqdm.tqdm(config.chr_names):
#     with open("%s/%s.pkl" % (config.out_dir, chromosome), 'rb') as chr_out:
#         out = pickle.load(chr_out)
#     phasing_result = phasing_result | out
#vcf_writer.write_phased_vcf(config.vcf, phasing_result, config.output_vcf)