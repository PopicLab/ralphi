import argparse
import torch
import time
import matplotlib
import utils.plotting as vis
import models.actor_critic as algs
import envs.phasing_env as envs
import seq.var as var
import seq.phased_vcf as vcf_writer
import subprocess
#matplotlib.use('macosx')

parser = argparse.ArgumentParser(description='Test haplotype phasing')
parser.add_argument('--model', default='../data/train/models/e_all_all_phasing_model.pt',
                    help='Pretrained model')
parser.add_argument('--panel', default="../data/test/frags/panel.txt",
                    help='Test fragment panel file')
parser.add_argument('--truth',
                    default="../../data/VCF/NA12878.ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf",
                    help='Ground truth VCF file')
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
        action = agent.select_action()
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

ground_truth_vcf = args.truth
output_vcf = ground_truth_vcf + '.latest.dphase.phased.vcf'
vcf_writer.write_phased_vcf(ground_truth_vcf, idx2var, output_vcf)
subprocess.call(["../third-party/HapCUT2/utilities/calculate_haplotype_statistics.py", "-v1", output_vcf, "-v2",
                 ground_truth_vcf], shell=True)

