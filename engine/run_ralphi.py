import torch
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
import subprocess
import random
import os, sys


def initialize(args, link):
    config_type = config_utils.CONFIG_TYPE.TEST
    if link:
         config_type = config_utils.CONFIG_TYPE.LINK
    config = config_utils.load_config(args.config, config_type=config_type)

    os.makedirs("%s/visualize/" % config.out_dir, exist_ok=True)
    random.seed(config.seed)
    torch.set_num_threads(config.num_cores_torch)
    torch.manual_seed(config.seed)
    logging.info("Running on %d processes" % config.n_procs)

    num_processes = min(config.n_procs, len(config.chr_names))
    chr_name_chunks = np.array_split(np.array(config.chr_names), num_processes)
    logging.info("Chromosomes/process partition: " + str([np.array2string(chk) for chk in chr_name_chunks]))
    Parallel(n_jobs=num_processes)(delayed(run_parallel)(chr_name_chunks[i], deepcopy(config)) for i in range(num_processes))

    logging.info("Merging results")
    chr_files = ["%s/%s.ralphi.vcf" % (config.out_dir, chromosome) for chromosome in config.chr_names]
    subprocess.run(['bcftools', 'concat', '-Oz', '-o', config.output_vcf] + chr_files)


def run_parallel(chr_names, config):  # runs phasing on the specified list of chromosomes
        logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=logging.getLevelName(config.logging_level),
                        handlers=[logging.FileHandler(f"{config.log_dir}/{'_'.join(chr_names)}.log", mode='w'),
                                  logging.StreamHandler(sys.stdout)],
                        force=True)

        for chromosome in chr_names:
            # -------- load variants and reads to generate input fragments
            config.fragments, config.variants = generate_fragments(config, chromosome)
            # -------- initialize the fragment graph environment and load the pre-trained agent
            env = envs.phasing_env.PhasingEnv(config, record_solutions=True)
            agent = models.actor_critic.DiscreteActorCriticAgent(env)
            agent.model.load_state_dict(torch.load(config.model))
            # -------- run ralphi on all the connected components of the fragment graph
            graph_count = 0
            while env.has_state():
                if env.state.frag_graph.trivial: env.process_error_free_instance()
                else:
                    agent.run_episode(config, test_mode=True)
                env.reset()
                graph_count += 1
            env.postprocess()

            # ------- output the phased VCF
            logging.info("Finished phasing, writing outputs for %s" % chromosome)
            if config.link:
                phase_set_remap = var.update_phase_sets(env.solutions)
                idx2var = var.apply_link(config.variants, phase_set_remap)
            else:
                idx2var = var.extract_variants(env.solutions)
                for v in idx2var.values():
                    v.assign_haplotype()
                idx2var = var.postprocess(env.solutions, idx2var, config)
            vcf_writer.write_phased_vcf(config.vcf, idx2var,
                                        "%s/%s.ralphi.vcf" % (config.out_dir, chromosome), chromosome)
            logging.info("Finished processing %s" % chromosome)
            