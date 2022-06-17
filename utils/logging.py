import csv

class Logger:
    def __init__(self, out_dir):
        # Set up some logging to help analyze agent behaviour as we train
        self.csv_stats_storage = open(out_dir + "/episodes_stats.csv", 'w')
        self.csv_writer = csv.writer(self.csv_stats_storage, delimiter=",")
        column_headers = ['Episode', 'Nodes', 'Edges', 'Reward', 'CutSize', 'ActorLoss', 'CriticLoss', 'SumLoss',
                          'Runtime']
        self.csv_writer.writerow(column_headers)

        self.validation_csv_stats_storage = open(out_dir + "/validation.csv", 'w')
        self.validation_csv_writer = csv.writer(self.validation_csv_stats_storage, delimiter=",")
        validation_column_headers = ['Episode', 'SumOfCuts', 'SumOfRewards']
        self.validation_csv_writer.writerow(validation_column_headers)

    def write_episode_stats(self, episode, g_num_nodes, g_num_edges, episode_reward, cut_size, actor_loss, critic_loss, sum_loss, time):
        self.csv_writer.writerow([episode, g_num_nodes, g_num_edges, episode_reward,
                                  cut_size, actor_loss, critic_loss, sum_loss, time])

    def write_validation_stats(self, episode, validation_sum_of_cuts, validation_sum_of_rewards):
        self.validation_csv_writer.writerow([episode, validation_sum_of_cuts, validation_sum_of_rewards])
        self.validation_csv_stats_storage.flush()