import argparse
import json
import pathlib
import pickle
from collections import defaultdict
from pprint import pprint

import torch
import torch.nn.functional as F
import wandb
import yaml
from easydict import EasyDict
from gin import GINModel
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.typing import SparseTensor
from tqdm import tqdm
from train_synthetic import count_parameters, set_seed


def get_ogbn_arxiv(as_sparse: bool = True):
    """
    see data description at https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
    """

    path_root = pathlib.Path(__file__).parent.resolve() / ".."
    if as_sparse:
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=path_root / "data", transform=ToSparseTensor())
    else:
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=path_root / "data")

    NUM_CLASSES = 40

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    graph = dataset[0]  # pyg graph object

    assert len(train_idx) + len(valid_idx) + len(test_idx) == graph.num_nodes
    assert graph.x.size(0) == graph.num_nodes
    assert torch.min(graph.y).item() == 0
    assert torch.max(graph.y).item() == NUM_CLASSES - 1

    return graph, train_idx, valid_idx, test_idx, NUM_CLASSES


def get_accuracy(logits, y, batch_size=None):
    """
    specify batch size if logits and y are too large
    """

    assert logits.size(0) == y.size(0)
    num_nodes = logits.size(0)

    if batch_size is None:

        y_hat = torch.argmax(logits, dim=-1, keepdim=True)
        acc = (y == y_hat).sum().item() / num_nodes

    else:

        assert batch_size <= num_nodes

        accs = []
        for batch in range(0, num_nodes, batch_size):
            logits_chunk = logits[batch : batch + batch_size]
            y_chunk = y[batch : batch + batch_size]
            y_hat_chunk = torch.argmax(logits_chunk, dim=-1, keepdim=True)
            accs.append((y_chunk == y_hat_chunk).sum().item())

        print("num batches", len(accs))
        acc = sum(accs) / num_nodes

    return acc


def train_model(
    data, model, mask, optimiser, loss_fn=torch.nn.functional.cross_entropy
):
    model.train()
    optimiser.zero_grad()
    y = data.y[mask]
    optimiser.zero_grad()
    logits = model(data)[mask]
    if y.ndim > 1:
        assert (y.ndim == 2) and (y.size(1) == 1)
        y = y.squeeze(dim=-1)
    loss = loss_fn(logits, y)
    loss.backward()
    optimiser.step()
    del logits
    torch.cuda.empty_cache()
    return loss.item()


@torch.no_grad()
def eval_model(data, model, mask):
    if sum(mask).item() == 0:
        return torch.nan
    model.eval()
    y = data.y[mask]
    logits = model(data)[mask]
    acc = get_accuracy(logits, y)
    del logits
    torch.cuda.empty_cache()
    return acc


def train_eval_loop(
    model,
    data,
    train_idx,
    val_idx,
    test_idx,
    # test_mask,
    lr: float,
    num_epochs: int,
    print_every: int,
    verbose: bool = False,
    log_wandb: bool = True,
):

    # optimiser and loss function
    # -------------------------------

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="mean"
    )  # xent loss for multiclass classification

    # dicts to store training stats
    # -------------------------------

    epoch2trainloss = {}
    epoch2trainacc = {}
    epoch2valacc = {}

    # initial metrics
    # -------------------------------

    train_acc = eval_model(data, model, train_idx)
    val_acc = eval_model(data, model, val_idx)
    # test_acc = eval_model(data, model, test_mask)

    epoch = 0
    epoch2trainacc[epoch] = train_acc
    epoch2valacc[epoch] = val_acc

    if log_wandb:
        wandb.log(
            {
                "train/acc": train_acc,
                "val/acc": val_acc,
                # "test/acc": test_acc,
            }
        )

    # training loop
    # -------------------------------

    print("Whole batch gradient descent on one graph...")

    with tqdm(range(1, num_epochs + 1), unit="e", disable=not verbose) as tepoch:
        # with tqdm(range(1,num_epochs+1), unit="e", disable=not verbose) as tepoch:
        for epoch in tepoch:

            train_loss = train_model(data, model, train_idx, optimiser, loss_fn)

            if log_wandb and not epoch % print_every == 0:
                wandb.log({"train/loss": train_loss})

            if epoch % print_every == 0:

                train_acc = eval_model(data, model, train_idx)
                val_acc = eval_model(data, model, val_idx)

                epoch2trainloss[epoch] = train_loss
                epoch2trainacc[epoch] = train_acc
                epoch2valacc[epoch] = val_acc

                if log_wandb:
                    wandb.log(
                        {
                            "train/loss": train_loss,
                            "train/acc": train_acc,
                            "val/acc": val_acc,
                        }
                    )

                tepoch.set_postfix(
                    train_loss=train_loss, train_acc=train_acc, val_acc=val_acc
                )

    # final metrics
    # -------------------------------

    train_acc = eval_model(data, model, train_idx)
    val_acc = eval_model(data, model, val_idx)
    # test_acc = eval_model(data, model, test_mask)

    epoch = num_epochs
    epoch2trainacc[epoch] = train_acc
    epoch2valacc[epoch] = val_acc

    if log_wandb:
        wandb.log(
            {
                "train/acc": train_acc,
                "val/acc": val_acc,
                # "test/acc": test_acc,
            }
        )

    # print metrics
    # -------------------------------

    print(
        f"Max val acc at epoch {max(epoch2valacc, key=epoch2valacc.get)}: {max(epoch2valacc.values())}"
    )
    print(f"Final val acc at epoch {num_epochs}: {epoch2valacc[num_epochs]}")

    if verbose:
        print("epoch: train loss | train acc | val acc")
        print(
            "\n".join(
                "{!r}: {:.5f} | {:.3f} | {:.3f}".format(
                    epoch,
                    epoch2trainloss[epoch],
                    epoch2trainacc[epoch],
                    epoch2valacc[epoch],
                )
                for epoch in epoch2trainloss  # this doesn't print the 0th epoch before training
            )
        )

    test_acc = eval_model(data, model, test_idx)

    del optimiser
    del loss_fn

    return test_acc


def get_rewire_edge_index(rewirer: str, as_sparse: bool = True):
    """
    these are various ways to instantiate Cayley clusters on this dataset
    """

    path_root = pathlib.Path(__file__).parent.resolve() / ".."
    base_rewire_dir = path_root / "data/arxiv-rewirings"

    if rewirer == "cayley":
        fn = "arxiv_rewire_by_cayley"
    elif rewirer == "class_all":
        fn = "arxiv_rewire_by_class_all"
    elif rewirer == "class_all_fully_connected_clusters":
        fn = "arxiv_rewire_by_class_all_fully_connected_clusters"
    elif rewirer == "class_train_only":
        fn = "arxiv_rewire_by_class_train_only"
    elif rewirer == "kmeans_all":
        fn = "arxiv_rewire_by_kmeans_all"
    elif rewirer == "mlp_all":
        fn = "arxiv_rewire_by_mlp_all"
    elif rewirer == "mlp_cs_all":
        fn = "arxiv_rewire_by_mlp_cs_all"
    elif rewirer == "mlp_feats_all":
        fn = "arxiv_rewire_by_mlp_feats_all"
    elif rewirer == "enriched-kmeans_all":
        fn = "arxiv_rewire_by_enriched-kmeans_all"
    elif rewirer == "knn":
        fn = "arxiv_rewire_by_knn"
    elif rewirer == "knn_mlp_feats":
        fn = "arxiv_rewire_by_knn_mlp_feats"
    else:
        raise NotImplementedError

    print(f"Using rewiring from: {fn}")

    with open(base_rewire_dir / fn, "rb") as f:
        rewire_edge_index = pickle.load(f)

    print("Opened")

    if as_sparse:
        rewire_edge_index = SparseTensor(
            row=rewire_edge_index[1],
            col=rewire_edge_index[0],
        )

    return rewire_edge_index


def main(config):

    pprint(config)

    # set device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # load dataset
    # -------------------------------
    graph, train_idx, valid_idx, test_idx, num_classes = get_ogbn_arxiv(as_sparse=config.train.as_sparse)

    print(
        "Reducing number of train samples by ",
        config.train.proportion_train_samples,
        "x",
    )
    num_train_samples = int(len(train_idx) * config.train.proportion_train_samples)
    permuted_indices = torch.randperm(len(train_idx))[:num_train_samples]
    train_idx = train_idx[permuted_indices]

    config.model.in_channels = graph.x.size(1)
    config.model.out_channels = num_classes

    # attach the rewirer
    # -------------------------------
    if config.model.rewirer is not None:
        graph.rewire_edge_index = get_rewire_edge_index(config.model.rewirer, as_sparse=config.train.as_sparse)

    # get moodel
    # -------------------------------

    set_seed(config.model.seed)

    model = GINModel(
        in_channels=graph.x.shape[1],
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers,
        out_channels=config.model.out_channels,
        drop_prob=config.model.drop_prob,
        only_original_graph=(config.model.approach == "only_original"),
        interleave_diff_graph=(config.model.approach == "interleave"),
        only_diff_graph=(config.model.approach == "only_diff_graph"),
        global_pool_aggr=None,
        norm=config.model.norm,
    )

    count_parameters(model)
    print(model)

    # train
    # -------------------------------

    model.to(device)
    graph.to(device)

    if config.train.log_wandb:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config,
            group=f"{config.train.proportion_train_samples}-rewirer-{config.model.rewirer}-{config.model.approach}",
        )
        wandb.run.name = f"{config.train.proportion_train_samples}-{config.model.rewirer}-{config.model.approach}-seed-{config.model.seed}"

    test_acc = train_eval_loop(
        model,
        graph,
        train_idx,
        valid_idx,
        test_idx,
        lr=config.train.lr,
        num_epochs=config.train.num_epochs,
        print_every=config.train.print_every,
        verbose=config.train.verbose,
        log_wandb=config.train.log_wandb,
    )

    results[config.train.proportion_train_samples][
        f"{config.model.rewirer}-{config.model.approach}"
    ].append(test_acc)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fn",
        default="../configs/ogbn-arxiv.yaml",
        help="configuration file name",
        type=str,
    )

    args = parser.parse_args()

    with open(args.config_fn, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    results = {x: defaultdict(list) for x in config.train.proportion_train_samples_list}

    for proportion_train_samples in config.train.proportion_train_samples_list:
        config.train.proportion_train_samples = proportion_train_samples
        for seed in config.model.seeds:
            config.model.seed = seed
            for approach in config.model.approaches:
                if approach == "only_original":
                    rewirers = [None]
                else:
                    rewirers = config.model.rewirers
                for rewirer in rewirers:
                    config.model.rewirer = rewirer
                    config.model.approach = approach
                    main(config)

    pprint(results)

    with open(
        "../data/results/arxiv_results.json",
        "w",
    ) as f:
        json.dump(dict(results), f, indent=4)
