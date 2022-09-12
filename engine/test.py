import argparse
import torch
import time
import utils.plotting as vis
import models.actor_critic as algs
import envs.phasing_env as envs
import seq.var as var
import seq.phased_vcf as vcf_writer
import subprocess
import utils.post_processing
import pickle
#matplotlib.use('macosx')

parser = argparse.ArgumentParser(description='Test haplotype phasing')
parser.add_argument('--model', help='Pretrained model')
parser.add_argument('--panel', help='Test fragment panel file')
parser.add_argument('--input_vcf', help='Input VCF file to phase')
parser.add_argument('--out_dir', help='output dir')
parser.add_argument('--num_cores', type=int, default=4, help='number of threads to use for Pytorch (default: 4)')

args = parser.parse_args()
torch.set_num_threads(args.num_cores)

env = envs.PhasingEnv(args.panel, record_solutions=True, skip_singleton_graphs=False, skip_trivial_graphs=False)
agent = algs.DiscreteActorCriticAgent(env)
agent.model.load_state_dict(torch.load(args.model))
n_episodes = 0

# run through all the components of the fragment graph
while env.has_state():
    if not env.state.frag_graph.trivial:
        # solve using exact algorithm
        print("component is error free")
        env.lookup_error_free_instance()
        n_episodes += 1
        env.reset()
        continue
    else:
        print("component has seq error")
    
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

with open(args.out_dir + "/phasing_output.pickle", 'wb') as phased_output:
     pickle.dump(env.solutions, phased_output, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved phasing output raw solutions to: ", args.out_dir + "/phasing_output.pickle")

# output the phased VCF (phase blocks)
idx2var = var.extract_variants(env.solutions)
for v in idx2var.values():
    v.assign_haplotype()
print("outputted phased VCF (phase blocks)")

idx2var = utils.post_processing.update_split_block_phase_sets(env.solutions, idx2var)
print("post-processed blocks that were split up due to ambiguous variants")

output_vcf = args.out_dir + "/dphase_phased.vcf"

vcf_writer.write_phased_vcf(args.input_vcf, idx2var, output_vcf)
print("Output written to: ", output_vcf)

# automatically call the evaluation script if ground truth is given
#subprocess.call(["../third-party/HapCUT2/utilities/calculate_haplotype_statistics.py", "-v1", args.output, "-v2",
#                 args.truth], shell=True)

