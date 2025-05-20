import networkx as nx
import unittest

from graphs import frag_graph


class TestSplitting(unittest.TestCase):
    def compare_lists(self, list1, list2):
        normalized_list1 = sorted([sorted(sublist) for sublist in list1])
        normalized_list2 = sorted([sorted(sublist) for sublist in list2])
        self.assertEqual(normalized_list1, normalized_list2)

    def test_two_arti(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 2)

        g.add_edge(3, 4)
        g.add_edge(3, 5)
        g.add_edge(5, 4)

        biconnected_components = [[1, 2, 3], [3, 4, 5]]
        articulation_points = [3]

        frag = frag_graph.FragGraph(g, [])
        arti, bic = frag.tarjan_algorithm()

        self.assertCountEqual(arti, articulation_points)
        self.compare_lists(bic, biconnected_components)

    def test_cycle(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 2)
        g.add_edge(3, 4)
        g.add_edge(4, 2)

        g.add_edge(5, 4)

        g.add_edge(5, 6)
        g.add_edge(7, 6)
        g.add_edge(5, 7)
        g.add_edge(7, 8)
        g.add_edge(6, 8)

        g.add_edge(7, 9)

        g.add_edge(11, 10)
        g.add_edge(9, 10)
        g.add_edge(9, 11)

        g.add_edge(10, 1)

        biconnected_components = [[i for i in range(1, 12)]]
        articulation_points = []

        frag = frag_graph.FragGraph(g, [])
        arti, bic = frag.tarjan_algorithm()
        self.assertCountEqual(arti, articulation_points)
        self.compare_lists(bic, biconnected_components)

    def test_three_arti(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 2)
        g.add_edge(3, 4)
        g.add_edge(4, 2)

        g.add_edge(5, 4)

        g.add_edge(5, 6)
        g.add_edge(7, 6)
        g.add_edge(5, 7)
        g.add_edge(7, 8)
        g.add_edge(6, 8)

        g.add_edge(7, 9)
        g.add_edge(7, 10)
        g.add_edge(9, 10)

        biconnected_components = [[1, 2, 3, 4], [4, 5], [6, 7, 8, 5], [7, 9, 10]]
        articulation_points = [4, 5, 7]

        frag = frag_graph.FragGraph(g, [])
        arti, bic = frag.tarjan_algorithm()

        self.assertCountEqual(arti, articulation_points)
        self.compare_lists(bic, biconnected_components)

    def test_one_arti_three_bic(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 2)
        g.add_edge(3, 4)
        g.add_edge(4, 2)

        g.add_edge(5, 4)
        g.add_edge(5, 6)
        g.add_edge(7, 5)
        g.add_edge(6, 7)
        g.add_edge(6, 4)

        g.add_edge(4, 8)
        g.add_edge(4, 9)
        g.add_edge(9, 8)

        biconnected_components = [[1, 2, 3, 4], [6, 7, 4, 5], [4, 9, 8]]
        articulation_points = [4]

        frag = frag_graph.FragGraph(g, [])
        arti, bic = frag.tarjan_algorithm()

        self.assertCountEqual(arti, articulation_points)
        self.compare_lists(bic, biconnected_components)

    def test_one_arti_start_three_bic(self):
        g = nx.Graph()
        g.add_edge(4, 2)
        g.add_edge(1, 3)
        g.add_edge(3, 2)
        g.add_edge(3, 4)
        g.add_edge(1, 2)

        g.add_edge(5, 4)
        g.add_edge(5, 6)
        g.add_edge(7, 5)
        g.add_edge(6, 7)
        g.add_edge(6, 4)

        g.add_edge(4, 8)
        g.add_edge(4, 9)
        g.add_edge(9, 8)

        biconnected_components = [[1, 2, 3, 4], [6, 7, 4, 5], [4, 9, 8]]
        articulation_points = [4]

        frag = frag_graph.FragGraph(g, [])
        arti, bic = frag.tarjan_algorithm()
        print(bic, arti)
        self.assertCountEqual(arti, articulation_points)
        self.compare_lists(bic, biconnected_components)

if __name__ == "__main__":
    unittest.main()