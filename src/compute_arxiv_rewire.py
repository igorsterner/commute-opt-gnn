import argparse
import pathlib
import pickle
import time

import einops
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from generate_cayley_graph import CayleyGraphGenerator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm
from train_arxiv_gin import get_ogbn_arxiv
from train_arxiv_mlp import MLP

path_root = pathlib.Path(__file__).parent.resolve() / ".."


def get_fully_connected_rewiring(colours, allowable_idx, num_colours: int):

    assert colours.ndim == 1, "Colours must be a one-dimensional array or tensor"

    colours_allowable = colours[allowable_idx]

    all_edges = []

    for colour in tqdm(range(num_colours), desc="Generating fully connected clusters"):
        nodes_of_colour = allowable_idx[colours_allowable == colour]
        edges_start, edges_end = torch.combinations(nodes_of_colour, 2).unbind(dim=1)
        all_edges.append(torch.stack([edges_start, edges_end], dim=0))

    edge_index = torch.cat(all_edges, dim=1)
    edge_index = torch.cat((edge_index, edge_index.flip(dims=(0,))), dim=-1)

    return edge_index


def get_cayley_clusters_rewiring(colours, allowable_idx, num_colours: int):
    """
    num_colours are the number of colours in label, assumed to be indexed from 0 to num_colours-1
    labels < 0 are ignored
    """

    assert colours.ndim == 1

    allowable_idx_set = set(allowable_idx.tolist())
    colours_allowable = colours[allowable_idx]
    map_allowable_to_all = {i: val for i, val in enumerate(allowable_idx.tolist())}

    for i in range(colours_allowable.size(0)):
        assert colours_allowable[i].item() == colours[map_allowable_to_all[i]].item()

    graph = nx.Graph()

    for colour in tqdm(range(num_colours), desc="Generating Cayley clusters"):

        num_nodes = (colours_allowable == colour).sum().item()

        print(colour, num_nodes)

        if num_nodes == 0:
            print("Concerning...")
            continue

        cg_gen = CayleyGraphGenerator(num_nodes)
        cg_gen.generate_cayley_graph()
        cg_gen.trim_graph()

        mapping = {
            org_idx: np.where(colours_allowable == colour)[0][col_idx]
            for org_idx, col_idx in zip(cg_gen.G_trimmed, range(num_nodes))
        }

        remapped_graph = nx.relabel_nodes(cg_gen.G_trimmed, mapping)

        graph = nx.union(graph, remapped_graph)

    # check valid!
    for edge in graph.edges:
        assert colours_allowable[edge[0]] == colours_allowable[edge[1]]
        assert edge[0] < len(colours_allowable)
        assert edge[1] < len(colours_allowable)

    # use original indices, not just indices into allowable
    graph_relabelled = nx.relabel_nodes(graph, map_allowable_to_all)

    # check valid!
    for edge in graph_relabelled.edges:
        assert colours[edge[0]] == colours[edge[1]]
        assert edge[0] in allowable_idx_set, f"{edge}"
        assert edge[1] in allowable_idx_set, f"{edge}"

    rewire_edge_index = einops.rearrange(
        torch.tensor(list(graph_relabelled.edges)),
        "e n -> n e",
    )

    # concatenate the reverse edges to ensure graph is undirected
    rewire_edge_index = torch.cat(
        (rewire_edge_index, rewire_edge_index.flip(dims=(0,))), dim=-1
    )

    return rewire_edge_index


def check_valid(rewire_edge_index, colours, allowable_idx):

    allowable_idx_set = set(allowable_idx.tolist())

    edges = [
        (rewire_edge_index[0, i].item(), rewire_edge_index[1, i].item())
        for i in range(rewire_edge_index.size(1))
    ]
    edges = set(edges)

    for edge in edges:
        node_a, node_b = edge
        assert (node_b, node_a) in edges, f"{edge}"
        assert colours[node_a].item() == colours[node_b].item()
        assert node_a in allowable_idx_set
        assert node_b in allowable_idx_set

    print("Passed basic checks!")


@torch.no_grad()
def get_colours_from_mlp(feats, device="cpu"):

    model = MLP(
        feats.size(-1), hidden_channels=256, out_channels=40, num_layers=3, dropout=0.5
    ).to(device)

    model.load_state_dict(
        torch.load(
            "../data/models/ogbn-arxiv-mlp-model.pth", map_location=torch.device("cpu")
        )
    )

    out = model(feats)

    print(out.argmax(dim=-1, keepdim=False).shape)

    return out.argmax(dim=-1, keepdim=False)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assign_colours_by",
        default="mlp_cs_all",
        help="Method for assigning colours for Cayley clusters",
        type=str,
    )
    args = parser.parse_args()

    assign_colours_by = args.assign_colours_by
    print(f"Computing rewiring according to: {assign_colours_by}")

    graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv()

    if assign_colours_by == "none":
        # with no colur assignments, construct Cayley expander over entire graph
        # this is agnostic to graph features and labels

        num_nodes = graph.num_nodes

        if num_nodes < 2:
            raise ValueError(
                "Cayley graph requires at least 2 nodes, but got only {}.".format(
                    num_nodes
                )
            )

        time_start = time.time()
        print("Instantiating CayleyGraphGenerator")
        cg_gen = CayleyGraphGenerator(num_nodes)
        print("Generating...")
        cg_gen.generate_cayley_graph()
        print("Trimming...")
        cg_gen.trim_graph()
        print("Done!")
        time_end = time.time()
        print(f"Time elapsed {time_end-time_start:.3f} s")

        rewired_graph = nx.relabel_nodes(
            cg_gen.G_trimmed, dict(zip(cg_gen.G_trimmed, range(num_nodes)))
        )

        rewire_edge_index = from_networkx(rewired_graph).edge_index

        with open("../data/arxiv-rewirings/arxiv_rewire_by_cayley", "wb") as f:
            pickle.dump(rewire_edge_index, f)

        return

    if assign_colours_by == "by_class_train":
        # Cayley clusters rewire edge index on only the train nodes
        # prevent leakage from val and test set labels into train

        rewire_edge_index = get_cayley_clusters_rewiring(
            graph.y.squeeze(),
            allowable_idx=train_idx,
            num_colours=num_classes,
        )

        with open(
            "../data/arxiv-rewirings/arxiv_rewire_by_class_train_only", "wb"
        ) as f:
            pickle.dump(rewire_edge_index, f)

        check_valid(rewire_edge_index, graph.y.squeeze(), train_idx)

        return

    if assign_colours_by == "by_class_all":
        # Cayley clusters rewire edge index on all nodes, irrespective of split
        # this is sensible only in the limit of perfect priors that are equivalent to knowing the label

        rewire_edge_index = get_cayley_clusters_rewiring(
            graph.y.squeeze(),
            allowable_idx=torch.tensor(range(graph.num_nodes)),
            num_colours=num_classes,
        )

        with open("../data/arxiv-rewirings/arxiv_rewire_by_class_all", "wb") as f:
            pickle.dump(rewire_edge_index, f)

        check_valid(
            rewire_edge_index, graph.y.squeeze(), torch.tensor(range(graph.num_nodes))
        )

        return

    if assign_colours_by == "mlp_all":

        mlp_colours = get_colours_from_mlp(graph.x)

        assert mlp_colours.size(0) == graph.num_nodes

        rewire_edge_index = get_cayley_clusters_rewiring(
            mlp_colours,
            allowable_idx=torch.tensor(range(graph.num_nodes)),
            num_colours=num_classes,
        )

        with open("../data/arxiv-rewirings/arxiv_rewire_by_mlp_all", "wb") as f:
            pickle.dump(rewire_edge_index, f)

        check_valid(
            rewire_edge_index, mlp_colours, torch.tensor(range(graph.num_nodes))
        )

        return


    if assign_colours_by == "by_class_all_fully_connected_clusters":
        # Cayley clusters rewire edge index on all nodes, irrespective of split
        # this is sensible only in the limit of perfect priors that are equivalent to knowing the label
        print("Generating rewire by class all fully connected clusters")

        rewire_edge_index = get_fully_connected_rewiring(
            graph.y.squeeze(),
            allowable_idx=torch.tensor(range(graph.num_nodes)),
            num_colours=num_classes,
        )

        with open(
            "../data/arxiv-rewirings/arxiv_rewire_by_class_all_fully_connected_clusters",
            "wb",
        ) as f:
            pickle.dump(rewire_edge_index, f)

        check_valid(
            rewire_edge_index, graph.y.squeeze(), torch.tensor(range(graph.num_nodes))
        )

        return

    raise NotImplementedError


if __name__ == "__main__":
    main()
