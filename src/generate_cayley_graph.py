from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix


class CayleyGraphGenerator:
    """
    Generate Cayley graphs for the special linear group SL(2, Z_n).
    """

    def __init__(self, V):
        """
        Initializes the class with a target number of vertices V.
        """
        self.V = V
        self.n = self._find_n()

        self.gen_set = [
            np.array([[1, 1], [0, 1]]),
            np.array([[1, 0], [1, 1]]),
        ]  # page 5
        self.dimension = self.gen_set[0].shape[0]

        self.adj_matrix = None
        self.G = None
        self.G_trimmed = None

    def _find_n(self):
        """
        Finds the smallest n such that the Cayley graph has at least V nodes.

        Equation (10)
        """
        n = 1
        while True:
            size = n**3 * np.prod([1 - 1 / p**2 for p in self._prime_factors(n)])
            if size >= self.V:
                return n
            n += 1

    def _prime_factors(self, n):
        """
        Returns a list of unique prime factors of n.
        """
        i = 2
        factors = set()
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.add(i)
        if n > 1:
            factors.add(n)
        return factors

    def generate_cayley_graph(self):
        """
        Generates and returns the adjacency matrix of the Cayley graph.

        Algorithm 1
        """
        if self.V == 1:
            self.adj_matrix = np.array([[0]])
            self.adj_matrix_to_graph()
        else:
            elements = self.get_group_elements()

            # self.adj_matrix = np.zeros((len(elements), len(elements)), dtype=int)
            self.adj_matrix = lil_matrix((len(elements), len(elements)), dtype=int)

            element_to_index = {
                self.matrix_to_tuple(element): idx
                for idx, element in enumerate(elements)
            }

            for i, u in enumerate(elements):
                for g in self.gen_set:
                    v = np.dot(u, g) % self.n
                    j = element_to_index[self.matrix_to_tuple(v)]
                    self.adj_matrix[i, j] = 1

            self.adj_matrix_to_graph()

    def get_group_elements(self):
        """
        Finds all elements of SL(2, Z_n).
        """
        elements = []
        for matrix in product(range(self.n), repeat=self.dimension**2):
            matrix = np.reshape(matrix, (2, 2)) % self.n
            if int(np.round(np.linalg.det(matrix))) % self.n == 1:
                elements.append(matrix)
        return elements

    def matrix_to_tuple(self, matrix):
        """
        Converts a numpy array matrix to a tuple of tuples.
        """
        return tuple(map(tuple, matrix))

    def adj_matrix_to_graph(self):
        """
        Creates a NetworkX graph from an adjacency matrix
        """

        self.G = nx.Graph(self.adj_matrix)

        # self.G_ = nx.Graph()

        # num_nodes = self.adj_matrix.shape[0]
        # self.G_.add_nodes_from(range(num_nodes))
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         if self.adj_matrix[i, j] == 1:
        #             self.G_.add_edge(i, j, weight=1)

        # from networkx.utils import graphs_equal
        # assert graphs_equal(self.G, self.G_)

        assert (
            self.G.number_of_nodes() >= self.V
        ), f"The graph has the wrong number of nodes, it has {self.G.number_of_nodes()} nodes but should have at least {self.V} nodes)"

    def trim_graph(self):
        """
        Performs BFS on the self.G graph and creates a new graph self.G_trimmed
        consisting of only the first self.V nodes visited

        Page 8
        """
        if self.G is None:
            raise Exception(
                "The graph G has not been created. Use create_cayley_graph() first."
            )

        try:
            bfs_nodes = list(nx.bfs_edges(self.G, source=0))
        except:
            raise Exception(
                f"Could not perform BFS starting at 0 on graph with nodes {self.G.nodes}"
            )

        visited_nodes = {0}
        for u, v in bfs_nodes:
            visited_nodes.add(v)
            if len(visited_nodes) >= self.V:
                break

        self.G_trimmed = self.G.subgraph(visited_nodes).copy()

    def visualize_graph(self, trimmed=True):
        """
        Visualizes the Cayley graph given an adjacency matrix.
        """
        if self.G is None:
            raise Exception(
                "You need to create the Cayley graph with create_cayley_graph() first"
            )
        elif trimmed and self.G_trimmed is None:
            raise Exception(
                "Great, you have your Cayley graph but now either set trimmed=False or run trim_graph()"
            )

        nx.draw(self.G_trimmed if trimmed else self.G, node_size=100)

        title = "Cayley Graph of $SL(2, \mathbb{Z}_" + str(self.n) + ")$"

        if trimmed:
            title += f", trimmed from {self.adj_matrix.shape[0]} to {self.V} nodes"

        plt.title(title)
        plt.show()
