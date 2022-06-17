import argparse
import torch
import time
import utils.plotting as vis
import models.actor_critic as algs
import envs.phasing_env as envs
import seq.var as var
import seq.phased_vcf as vcf_writer
import subprocess
#matplotlib.use('macosx')

parser = argparse.ArgumentParser(description='Test haplotype phasing')
parser.add_argument('--model', help='Pretrained model')
parser.add_argument('--panel', help='Test fragment panel file')
parser.add_argument('--input_vcf', help='Input VCF file to phase')
parser.add_argument('--output_vcf', help='Output VCF file')
args = parser.parse_args()

env = envs.PhasingEnv(args.panel, record_solutions=True, skip_singleton_graphs=False)
agent = algs.DiscreteActorCriticAgent(env)
agent.model.load_state_dict(torch.load(args.model))
n_episodes = 0

# run through all the components of the fragment graph
while env.has_state():
    start_time = time.time()
    done = False
    while not done:
        action = agent.select_action(greedy=True)
        _, _, done = env.step(action)
    end_time = time.time()
    #print("Runtime: ", end_time - start_time)
    # node_labels = env.state.g.ndata['x'][:, 0].cpu().squeeze().numpy().tolist()
    # vis.plot_network(env.state.g.to_networkx(), node_labels)
    # print("Accuracy: ", graphs.eval_assignment(node_labels, env.state.frag_graph.node_id2hap_id))
    env.reset()
    n_episodes += 1

print("NUM EPISODES: ", n_episodes)
# output the phased VCF (phase blocks)
idx2var = var.extract_variants(env.solutions)
for v in idx2var.values():
    v.assign_haplotype()
vcf_writer.write_phased_vcf(args.input_vcf, idx2var, args.output_vcf)
print("Output written to: ", args.output_vcf)

# automatically call the evaluation script if ground truth is given
#subprocess.call(["../third-party/HapCUT2/utilities/calculate_haplotype_statistics.py", "-v1", args.output, "-v2",
#                 args.truth], shell=True)

