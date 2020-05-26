import torch
import time
import matplotlib
import utils.plotting as vis
import models.actor_critic as algs
import envs.phasing_env as envs
import graphs.frag_graph as graphs
matplotlib.use('Agg')

MODEL_PATH = 'phasing_model.pt'
env = envs.PhasingEnv()
agent = algs.DiscreteActorCriticAgent(env)
agent.model.load_state_dict(torch.load(MODEL_PATH))

env.reset()
start_time = time.time()
done = False
while not done:
    action = agent.select_action()
    _, _, done = env.step(action)
end_time = time.time()
node_labels = env.state.g.ndata['x'][:, 0].cpu().squeeze().numpy().tolist()
vis.plot_network(env.state.g.to_networkx(), node_labels)
print("Runtime: ", end_time - start_time)
print("Accuracy: ", graphs.eval_assignment(node_labels, env.state.haplotype_graph.node_id2hap_id))

# TF
# pi = tf.nn.softmax(pi, axis=0)
# action = tf.random.categorical(pi, 1)[0][0].numpy()
