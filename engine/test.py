import torch
import time
import matplotlib
import utils.plotting as vis
import models.actor_critic as algs
import envs.phasing_env as envs
import seq.var as var
import seq.phased_vcf as vcf_writer
matplotlib.use('macosx')

MODEL_PATH = 'phasing_model.pt'
test_panel_file = "../data/test/frags/panel.txt"
env = envs.PhasingEnv(test_panel_file, record_solutions=True)
agent = algs.DiscreteActorCriticAgent(env)
agent.model.load_state_dict(torch.load(MODEL_PATH))

# run through all the components of the fragment graph
while env.has_state():
    start_time = time.time()
    done = False
    while not done:
        action = agent.select_action()
        _, _, done = env.step(action)
    end_time = time.time()
    print("Runtime: ", end_time - start_time)
    # node_labels = env.state.g.ndata['x'][:, 0].cpu().squeeze().numpy().tolist()
    # vis.plot_network(env.state.g.to_networkx(), node_labels)
    # print("Accuracy: ", graphs.eval_assignment(node_labels, env.state.frag_graph.node_id2hap_id))
    env.reset()

# output the phased VCF (phase blocks)
idx2var = var.extract_variants(env.solutions)
for v in idx2var.values():
    v.assign_haplotype()

vcf_writer.write_phased_vcf("/Users/vpopic/research/data/VCF/"
                            "NA12878.ALL.chr20.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf", idx2var)


