import argparse
import torch

import models.actor_critic
import envs.phasing_env
import seq.var as var
import engine.config as config_utils
import seq.phased_vcf as vcf_writer


# ------ CLI ------
print("*********************************")
print("*  ralphi: haplotype assembly   *")  # TODO: add ralphi version
print("*********************************")
parser = argparse.ArgumentParser(description='Run ralphi haplotype assembly')
parser.add_argument('--config', help='YAML with parameters')
args = parser.parse_args()
# -----------------

config = config_utils.load_config(args.config, config_type=config_utils.CONFIG_TYPE.TEST)
torch.set_num_threads(config.num_cores_torch)

# -------- initialize the fragment graph environment and RL agent
env = envs.phasing_env.PhasingEnv(config, record_solutions=True)
agent = models.actor_critic.DiscreteActorCriticAgent(env)
agent.model.load_state_dict(torch.load(config.model))

# -------- run ralphi on all the connected components of the fragment graph
while env.has_state():
    if env.state.frag_graph.trivial: env.process_error_free_instance()
    else: agent.run_episode(config, test_mode=True)
    env.reset()

# ------- output the phased VCF
idx2var = var.extract_variants(env.solutions)
for v in idx2var.values():
    v.assign_haplotype()
vcf_writer.write_phased_vcf(config.input_vcf, idx2var, config.output_vcf)

