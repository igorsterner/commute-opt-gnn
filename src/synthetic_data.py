import random
from functools import partial

import einops
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils.convert import from_networkx

from generate_cayley_graph import CayleyGraphGenerator


class MyGraph:
    def __init__(self, id, data):
        self.id = id
        self.data = data
        self.rewire_edge_index = None

    @property
    def num_nodes(self):
        return self.data.num_nodes

    @property
    def num_edges(self):
        return self.data.num_edges

    @property
    def edge_index(self):
        return self.data.edge_index

    @property
    def dim_feat(self):
        return self.data.x.shape[1]

    @property
    def x(self):
        return self.data.x

    @property
    def y(self):
        return self.data.y

    @property
    def edge_attr(self):
        return self.data.edge_attr

    def draw(self, layout=partial(nx.spring_layout, seed=42), with_labels=True):
        g = to_networkx(self.data, to_undirected=True)
        return nx.draw(g, pos=layout(g), with_labels=with_labels)

    def draw_rewire(self, layout=partial(nx.spring_layout, seed=42), with_labels=True):
        rewire_edge_index_tuples = [
            (edge[0].item(), edge[1].item()) for edge in self.rewire_edge_index.T
        ]
        g = nx.Graph(rewire_edge_index_tuples)
        return nx.draw(g, pos=layout(g), with_labels=with_labels)

    def attach_cayley(self):
        if self.num_nodes < 2:
            raise ValueError(
                "Cayley graph requires at least 2 nodes, but got only {}.".format(
                    self.num_nodes
                )
            )
        cg_gen = CayleyGraphGenerator(self.num_nodes)
        cg_gen.generate_cayley_graph()
        cg_gen.trim_graph()

        rewired_graph = nx.relabel_nodes(
            cg_gen.G_trimmed, dict(zip(cg_gen.G_trimmed, range(self.num_nodes)))
        )

        # rewire_edge_index = einops.rearrange(
        #     torch.tensor(list(rewire_edge_index.edges)),
        #     "e n -> n e",
        # )

        rewire_edge_index = from_networkx(rewired_graph).edge_index

        self.rewire_edge_index = rewire_edge_index

    def attach_fully_connected(self):
        rewire_edge_index = [
            (i, j)
            for i in range(self.num_nodes)
            for j in range(self.num_nodes)
            if i != j
        ]

        rewire_edge_index = einops.rearrange(
            torch.tensor(rewire_edge_index), "e n -> n e"
        )

        self.rewire_edge_index = rewire_edge_index


class ColourInteract(MyGraph):
    def __init__(
        self,
        id,
        data,
        c1,
        c2,
        num_colours,
        mean_num_nodes,
        normalise,
        distances=None,
        x=None,
        y=None,
        colours=None,
    ):
        super().__init__(id, data)

        self.c1 = c1
        self.c2 = c2
        self.num_colours = num_colours

        self.num_coloured_nodes = torch.randint(low=25, high=75, size=(1,)).item()

        self.mean_num_nodes = mean_num_nodes

        self.values = None
        self.colours = colours

        self.normalise = normalise

        self.distances = distances

        self._set_x(x)
        self._set_y(y)

    def _set_values(self):
        """
        set one-dimensional "values", drawn uniformly from the unit interval
        """
        num_nodes, dim_feat = self.num_nodes, 1
        values = torch.rand((num_nodes, dim_feat))
        self.values = values

    def _set_colours(self):
        """
        set categorical "colours", uniformly from a set of self.size num_colours
        """
        colours = torch.randint(low=0, high=self.num_colours, size=(self.num_nodes,))

        one_hot_colours = torch.nn.functional.one_hot(
                colours, num_classes=self.num_colours
            )
        try:
            indices_to_keep_coloured = random.sample(range(self.num_nodes), self.num_coloured_nodes)
            indices_to_make_uncoloured = [idx for idx in range(self.num_nodes) if idx not in indices_to_keep_coloured]
        except:
            raise Exception(f"Tried to sample {self.num_coloured_nodes} nodes from {self.num_nodes} nodes")
        
        colours[indices_to_make_uncoloured] = -1
        one_hot_colours[indices_to_make_uncoloured] = torch.zeros(self.num_colours, dtype=torch.int64)

        assert sum(torch.sum(one_hot_colours, dim=0)) == self.num_coloured_nodes

        self.one_hot_colours = one_hot_colours
        self.colours = colours

    def _set_x(self, x):
        """
        set features as the concatenation of "values" and "colours"
        """
        if x is None:
            if self.values is None:
                self._set_values()
            if self.colours is None:
                self._set_colours()

            x = torch.hstack((self.values, self.one_hot_colours))
            assert x.shape == (self.num_nodes, 1 + self.num_colours)

        self.data.x = x

    def _set_y(self, y):
        """
        Compute the regression target y for the graph as:
            \sum_{d} \sum_{i,j at distance 1} c_1 exp(v_i + v_j)
            +
            \sum_{c \in colours} \sum_{i,j of colour c} c_2 exp(v_i + v_j)
        where v_i is the value associated with node i
        Excluding self-interactions (i.e. nodes separated by distance 0).
        Sum runs over distinct pairs only (i.e. (i,j) and (j,i) are not double-counted).
        """

        if y is None:
            g = to_networkx(self.data, to_undirected=True)

            if self.distances is None:
                self.distances = dict(nx.all_pairs_shortest_path_length(g))

            dist_matrix = np.zeros((self.num_nodes, self.num_nodes))

            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):  # Avoid self-interactions
                    dist_matrix[i, j] = self.distances[i][j]

            dist_matrix = torch.tensor(dist_matrix)
            mask_1 = (dist_matrix == 1)

            interactions = torch.exp(self.values + self.values.T)
            interactions.fill_diagonal_(0)
            interactions = torch.triu(interactions)

            y = torch.sum(self.c1 * interactions[mask_1])

            colour_interactions = 0

            for color in range(self.num_colours):
                color_mask = (self.colours == color)

                same_color_matrix = color_mask.unsqueeze(1) & color_mask.unsqueeze(0)
                same_color_matrix.fill_diagonal_(0)
                same_color_matrix = torch.triu(same_color_matrix)

                colour_interactions += torch.sum(same_color_matrix)

                y += torch.sum(self.c2 * interactions[same_color_matrix])


            # print(f"Contribution from dist 1 interactions: ", torch.sum(self.c1 * interactions[mask_1]))
            # print(f"Contributions from colour interactions: ", y - torch.sum(self.c1 * interactions[mask_1]))

            if self.normalise:
                y = y / (self.num_nodes)
            # else:
            #     y = y / 1000
                # y = y / (100 ** 2)


            # print(colour_interactions)


            # print(y)
            # y = y / self.mean_num_nodes
            y = torch.tensor([y], dtype=torch.float)
            # y = y / (self.c2 / self.c1)
            # y = (1+torch.sin(y))/2
            # y = torch.tanh(y)

        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None

    def attach_rewirer(self, rewirer):
        if rewirer is None:
            pass
        elif rewirer == "cayley":
            self.attach_cayley()
        elif rewirer == "fully_connected":
            self.attach_fully_connected()
        elif rewirer == "cayley_clusters":
            self.attach_cayley_clusters()
        elif rewirer == "unconnected_cayley_clusters":
            self.attach_cayley_clusters(connect_clusters=False)
        elif rewirer == "fully_connected_clusters":
            self.attach_fully_connected_clusters()
        else:
            raise NotImplementedError

    def to_torch_data(self):
        """
        casts to the usual torch Data object
        """
        return Data(
            x=self.x,
            y=self.y,
            edge_index=self.edge_index,
            rewire_edge_index=self.rewire_edge_index,
            id=self.id,
        )

    def attach_fully_connected_clusters(self):
        graph = nx.Graph()

        for colour in range(-1, self.num_colours):
            num_nodes = (self.colours == colour).sum().item()
            this_colour_idxs = [idx for idx in range(self.num_nodes) if self.colours[idx] == colour]

            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    graph.add_edge(this_colour_idxs[i], this_colour_idxs[j])

        graph_sorted = nx.Graph()
        graph_sorted.add_nodes_from(sorted(graph.nodes(data=True)))
        graph_sorted.add_edges_from(graph.edges(data=True))

        # nx.draw(graph_sorted, node_color=self.colours, node_size=50)
        # plt.savefig("test-fc.png")
        # exit()

        # rewire_edge_index = einops.rearrange(
        #     torch.tensor(list(graph_sorted.edges)),
        #     "e n -> n e",
        # )

        rewire_edge_index = from_networkx(graph_sorted).edge_index

        self.rewire_edge_index = rewire_edge_index

    def attach_cayley_clusters(self, connect_clusters=True):
        graph = nx.Graph()

        for colour in range(-1, self.num_colours):
            num_nodes = (self.colours == colour).sum().item()

            if num_nodes == 0:
                continue

            cg_gen = CayleyGraphGenerator(num_nodes)
            cg_gen.generate_cayley_graph()
            cg_gen.trim_graph()

            assert nx.is_connected(cg_gen.G_trimmed)

            mapping = {
                org_idx: np.where(self.colours == colour)[0][col_idx]
                for org_idx, col_idx in zip(cg_gen.G_trimmed.nodes, range(num_nodes))
            }
            
            remapped_graph = nx.relabel_nodes(cg_gen.G_trimmed, mapping)

            assert nx.is_connected(remapped_graph)

            if connect_clusters and colour >= 0:
                node1 = random.choice(list(graph.nodes))
                node2 = random.choice(list(remapped_graph.nodes))

            graph = nx.union(graph, remapped_graph)

            if connect_clusters and colour >= 0:
                graph.add_edge(node1, node2)


        graph_sorted = nx.Graph()
        graph_sorted.add_nodes_from(sorted(graph.nodes(data=True)))
        graph_sorted.add_edges_from(graph.edges(data=True))

        # reverse_edges = [(v, u) for (u, v) in graph.edges()]
        # graph_sorted.add_edges_from(reverse_edges)

        # # if self.num_nodes > 160:
        # colours = {-1: 'white', 0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'orange', 6: 'pink', 7: 'brown', 8: 'grey', 9: 'cyan'}
        # node_colours = [colours[int(colour)] for colour in self.colours]
        # plt.clf()
        # nx.draw(graph_sorted, node_color=node_colours, node_size=50)
        # plt.savefig("test-cayley-clusters.png")
        # exit()

        # def draw_aligned_graphs(g1, g2, layout=nx.spring_layout, shift_x=2.0, filename='aligned-graphs.png'):
        #     # Compute layouts
        #     pos_g1 = layout(g1)
        #     # Adjust the layout for g2 by shifting all positions from g1 layout
        #     pos_g2 = {node: (x + shift_x, y) for node, (x, y) in layout(g2).items()}

        #     # Figure setup
        #     plt.figure(figsize=(12, 8))
        #     ax = plt.gca()

        #     # Draw the first graph
        #     nx.draw(g1, pos=pos_g1, node_color=node_colours, edge_color='k', ax=ax, node_size=50, edgecolors ='black')

        #     # Draw the second graph, shifted
        #     nx.draw(g2, pos=pos_g2, node_color=node_colours, edge_color='k', ax=ax, node_size=50, edgecolors ='black')

        #     # Draw dashed lines between corresponding nodes
        #     for node in g1.nodes():
        #         if node in pos_g1 and node in pos_g2:  # Ensure node exists in both layouts
        #             start_pos = pos_g1[node]
        #             end_pos = pos_g2[node]
        #             plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], "k--", linewidth=0.5, zorder=-1, color='gray', alpha=0.5)

        #     plt.axis('equal')
        #     plt.axis('off')  # Optionally turn off the axis for a cleaner look
        #     plt.savefig(filename, bbox_inches='tight')
        #     plt.show()

        # draw_aligned_graphs(to_networkx(self.data, to_undirected=True), graph_sorted)

        rewire_edge_index = from_networkx(graph_sorted).edge_index

        self.rewire_edge_index = rewire_edge_index


class SalientDists(MyGraph):
    def __init__(
        self,
        id,
        data,
        c1,
        c2,
        c3,
        d,
        distances=None,
        x=None,
        y=None,
        seed=42,
        rewirer=None,
        normalise=True,
    ):
        super().__init__(id, data)

        torch.manual_seed(seed)
        random.seed(seed)

        self.c1, self.c2, self.c3, self.d = c1, c2, c3, d
        self.normalise = normalise

        self.distances = distances

        self._set_x(x)
        self._set_y(y)
        self.attach_rewirer(rewirer)

    def _set_x(self, x):
        """
        set one-dimensional features, drawn uniformly from the unit interval
        """
        if x is None:
            num_nodes, dim_feat = self.num_nodes, 1
            x = torch.rand((num_nodes, dim_feat))

        self.data.x = x

    def _set_y(self, y):
        """
        Compute the regression target y for the graph as:
            \sum_{i,j at distance 1} c_1 exp(x_i + x_j)
            +
            \sum_{i,j at distance d} c_2 exp(x_i + x_j)
            +
            \sum_{i,j at distance other} c_3 exp(x_i + x_j)
        Excluding self-interactions (i.e. nodes separated by distance 0).
        Sum runs over distinct pairs only (i.e. (i,j) and (j,i) are not double-counted).
        """
        if y is None:
            g = to_networkx(self.data, to_undirected=True)

            if self.distances is None:
                distances = dict(nx.all_pairs_shortest_path_length(g))
                self.distances = distances

            dist_matrix = np.zeros((self.num_nodes, self.num_nodes))

            for i in range(self.num_nodes):
                for j in range(
                    i + 1, self.num_nodes
                ):  # we choose to avoid self-interactions
                    dist_matrix[i, j] = self.distances[i][j]

            dist_matrix = torch.tensor(dist_matrix)

            sum_x = self.data.x + self.data.x.T

            mask_1 = dist_matrix == 1
            mask_d = dist_matrix == self.d
            mask_other = (
                (dist_matrix != 1) & (dist_matrix != self.d) & (dist_matrix != 0)
            )

            y = (
                torch.sum(self.c1 * torch.exp(sum_x[mask_1]))
                + torch.sum(self.c2 * torch.exp(sum_x[mask_d]))
                + torch.sum(self.c3 * torch.exp(sum_x[mask_other]))
            )

            if self.normalise:
                y = y / self.num_nodes
            y = torch.tensor([y], dtype=torch.float)

        self.data.y = y

    def _set_edge_attr(self):
        """
        we are not concerned with edge attributes in this synthetic dataset
        """
        self.data.edge_attr = None

    def attach_rewirer(self, rewirer):
        if rewirer is None:
            pass

        elif rewirer == "cayley":
            self.attach_cayley()

        elif rewirer == "interacting_pairs":
            rewire_edge_index = []

            for i in self.distances:
                for j, dist in self.distances[i].items():
                    if (dist == 1) or (dist == self.d):
                        rewire_edge_index.append((i, j))

            rewire_edge_index = einops.rearrange(
                torch.tensor(rewire_edge_index), "e n -> n e"
            )
            self.rewire_edge_index = rewire_edge_index

        elif rewirer == "distance_d_pairs":
            rewire_edge_index = []

            for i in self.distances:
                for j, dist in self.distances[i].items():
                    if dist == self.d:
                        rewire_edge_index.append((i, j))

            rewire_edge_index = einops.rearrange(
                torch.tensor(rewire_edge_index), "e n -> n e"
            )
            self.rewire_edge_index = rewire_edge_index

        elif rewirer == "aligned_cayley":
            # create G1 based on distance d pairs
            g1 = nx.Graph()
            for i in self.distances:
                g1.add_node(i)
                for j, dist in self.distances[i].items():
                    if dist == self.d:
                        g1.add_edge(i, j)

            # print(f"{g1.nodes=}")
            # print(f"{g1.edges=}")
            # create G2 as trimmed Cayley expander

            if self.num_nodes < 2:
                raise ValueError(
                    "Cayley graph requires at least 2 nodes, but got only {}.".format(
                        self.num_nodes
                    )
                )

            cg_gen = CayleyGraphGenerator(self.num_nodes)
            cg_gen.generate_cayley_graph()
            cg_gen.trim_graph()
            g2 = cg_gen.G_trimmed

            # get correspondence between nodes
            correspondence_1_to_2 = get_correspondence(g1, g2)
            correspondence_2_to_1 = {v: k for k, v in correspondence_1_to_2.items()}

            # print(correspondence_2_to_1)

            # print(f"{g2.nodes=}")
            # print(f"{g2.edges=}")

            # produce rewired edge index

            g2_aligned = nx.relabel_nodes(g2, correspondence_2_to_1)

            # print(f"{g2_aligned.nodes=}")
            # print(f"{g2_aligned.edges=}")

            # sort this graph to get correctly labelled edge_index using from_networkx
            g2_aligned_sorted = nx.Graph()
            g2_aligned_sorted.add_nodes_from(sorted(g2_aligned.nodes(data=True)))
            g2_aligned_sorted.add_edges_from(g2_aligned.edges(data=True))

            # print(f"{g2_aligned_sorted.nodes=}")
            # print(f"{g2_aligned_sorted.edges=}")

            rewire_edge_index = from_networkx(g2_aligned_sorted).edge_index
            
            # rewire_edge_index_alt = einops.rearrange(
            #     torch.tensor(list(g2_aligned.edges)),
            #     "e n -> n e",
            # )

            # print(rewire_edge_index)

            self.rewire_edge_index = rewire_edge_index

        elif rewirer == "fully_connected":
            self.attach_fully_connected()

        else:
            raise NotImplementedError

    def to_torch_data(self):
        """
        casts to the usual torch Data object
        """
        return Data(
            x=self.x,
            y=self.y,
            edge_index=self.edge_index,
            rewire_edge_index=self.rewire_edge_index,
            id=self.id,
        )


def count_captured_edges(g1, g2, correspondence):
    """
    Count the fraction of edges in g1 included in g2, if nodes are mapped according to correspondence
    Success metric for graph alignment step...
    """
    cnt = 0
    for edge in g1.edges:
        edge_c = (correspondence[edge[0]], correspondence[edge[1]])
        if edge_c in g2.edges:
            cnt += 1
    return cnt / g1.number_of_edges()


def get_correspondence(g1, g2):
    """
    Naive graph alignment between two equal size graphs
    Returns a correspondence dictionary mapping each g1 nodes to g2 nodes bijectively
    """

    assert g1.number_of_nodes() == g2.number_of_nodes()

    correspondence = {node: None for node in list(g1.nodes)}
    g2_nodes_remaining = set(g2.nodes)

    for node1 in correspondence:
        if correspondence[node1] is not None:
            continue

        node2 = g2_nodes_remaining.pop()
        correspondence[node1] = node2
        neighbs = list(g1.neighbors(node1))

        for neighb in neighbs:
            if correspondence[neighb] is not None:
                continue
            candidates = list(
                node for node in g2.neighbors(node2) if node in g2_nodes_remaining
            )
            if len(candidates) == 0:
                continue
            cand = candidates[0]
            g2_nodes_remaining.remove(cand)
            correspondence[neighb] = cand

    return correspondence
