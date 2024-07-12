import argparse
import json
import math
import os
import random
from pprint import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict
from prettytable import PrettyTable
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from generate_synthetic_data import get_data_ColourInteract, get_data_SalientDists
from gin import GINModel

device = torch.device("cpu")


def train_model(data, model, optimiser, loss_fn):
    model.train()
    optimiser.zero_grad()
    y_pred = model(data)
    loss = loss_fn(y_pred, data.y)
    loss.backward()
    optimiser.step()
    return loss.item()


@torch.no_grad()
def eval_model(data_loader, model, loss_fn):
    model.eval()
    loss = []
    for data in data_loader:
        y_pred = model(data)
        loss.append(loss_fn(y_pred, data.y).item())
    return np.mean(loss)


def train_eval_loop(
    model,
    data_loader_train,
    data_loader_val,
    lr: float,
    num_epochs: int,
    print_every,
    loss_fn,
    gamma_decay,
    warmup,
    train_std,
    verbose=False,
    log_wandb=True,
):
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    warmup_scheduler = LinearLR(optimiser, start_factor=1e-4, total_iters=warmup)
    decay_scheduler = ExponentialLR(optimiser, gamma=gamma_decay)

    scheduler = SequentialLR(
        optimiser, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup]
    )

    if loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss(reduction="mean")
        if verbose:
            print("Using MSE Loss...")
    elif loss_fn == "CEL":
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        if verbose:
            print("Using Cross Entropy Loss...")

    epoch2trainloss = {}
    epoch2valloss = {}

    with tqdm(range(num_epochs), unit="e", disable=not verbose) as tepoch:
        for epoch in tepoch:
            running_train_loss = []
            tepoch.set_description(f"Epoch {epoch}")
            iters = len(data_loader_train)
            for i, data_train in enumerate(data_loader_train):
                running_train_loss.append(
                    train_model(data_train, model, optimiser, loss_fn)
                )
            scheduler.step()

            train_loss = np.mean(running_train_loss)

            if epoch % print_every == 0:
                val_loss = eval_model(data_loader_val, model, loss_fn)
                tepoch.set_postfix(train_loss=train_loss, val_loss=val_loss)
                epoch2trainloss[epoch] = train_loss
                epoch2valloss[epoch] = val_loss
                if log_wandb:
                    wandb.log(
                        {
                            "train/loss": train_loss,
                            "train/loss_over_variance": train_loss / (train_std**2),
                            "eval/loss": val_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        }
                    )
            else:
                if log_wandb:
                    wandb.log(
                        {
                            "train/loss": train_loss,
                            "train/loss_over_variance": train_loss / (train_std**2),
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        }
                    )

    epoch2valloss[num_epochs] = eval_model(data_loader_val, model, loss_fn)
    wandb.log({"eval/loss": epoch2valloss[num_epochs]})

    print(
        f"Minimum val loss at epoch {min(epoch2valloss, key=epoch2valloss.get)}: {min(epoch2valloss.values())}"
    )

    print(f"Final val loss at epoch {num_epochs}: {epoch2valloss[num_epochs]}")

    if verbose:
        print("Train / Val loss by epoch")
        print(
            "\n".join(
                "{!r}: {:.7f} / {:.7f}".format(
                    epoch, epoch2trainloss[epoch], epoch2valloss[epoch]
                )
                for epoch in epoch2trainloss
            )
        )

    end_results = {
        "end": val_loss,
        "best": min(epoch2valloss.values()),
        "train/loss_over_variance": train_loss / (train_std**2),
    }

    del optimiser
    del loss_fn

    return end_results


def quick_run(rewirers, config_file="../configs/debug_ColourInteract.yaml"):

    with open(config_file, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    pprint(config)

    # Get data
    set_seed(config.data.seed)

    if config.data.name == "SalientDists":
        graphs_train, graphs_val = get_data_SalientDists(
            dataset=config.data.dataset,
            device=device,
            c1=config.data.c1,
            c2=config.data.c2,
            c3=config.data.c3,
            d=config.data.d,
            normalise=config.data.normalise,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            min_val_nodes=config.data.min_val_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            verbose=config.run.silent,
        )
    elif config.data.name == "ColourInteract":
        config.data.c1 = 1 / (1 + config.data.c2_over_c1)
        config.data.c2 = config.data.c2_over_c1 / (1 + config.data.c2_over_c1)

        assert math.isclose(
            config.data.c1 + config.data.c2, 1.0
        ), f"{config.data.c1} + {config.data.c2} != 1.0"

        graphs_train, graphs_val = get_data_ColourInteract(
            dataset=config.data.dataset,
            c1=config.data.c1,
            c2=config.data.c2,
            normalise=config.data.normalise,
            num_colours=config.data.num_colours,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            min_val_nodes=config.data.min_val_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            verbose=config.run.silent,
        )
    else:
        raise NotImplementedError

    train_mean = np.mean([g.y.cpu() for g in graphs_train])
    train_std = np.std([g.y.cpu() for g in graphs_train])
    print(f"train targets: {train_mean:.2f} +/- {train_std:.3f}")
    print(
        f"val targets: {np.mean([g.y.cpu() for g in graphs_val]):.2f} +/- {np.std([g.y.cpu() for g in graphs_val]):.3f}"
    )

    set_seed(config.model.seed)

    graphs_train_base = []
    for g in graphs_train:
        graphs_train_base.append(g.to_torch_data().to(device))

    graphs_val_base = []
    for g in graphs_val:
        graphs_val_base.append(g.to_torch_data().to(device))

    dl_train = DataLoader(graphs_train_base, batch_size=config.train.train_batch_size)
    dl_val = DataLoader(graphs_val_base, batch_size=config.train.val_batch_size)

    in_channels = graphs_train[0].x.shape[1]
    out_channels = (
        1 if len(graphs_train[0].y.shape) == 0 else len(graphs_train[0].y.shape)
    )

    model = GINModel(
        in_channels=in_channels,
        hidden_channels=config.model.hidden_channels,
        num_layers=config.model.num_layers,
        out_channels=out_channels,
        drop_prob=config.model.drop_prob,
        only_original_graph=True,
        global_pool_aggr=config.model.global_pool_aggr,
        norm=config.model.norm,
    )
    model.to(device)

    end_results = train_eval_loop(
        model,
        dl_train,
        dl_val,
        lr=config.train.lr,
        num_epochs=config.train.num_epochs,
        print_every=config.train.print_every,
        verbose=not config.run.silent,
        log_wandb=False,
        loss_fn=config.train.loss_fn,
        gamma_decay=config.train.gamma_decay,
        warmup=config.train.warmup,
        train_std=train_std,
    )

    for num, rewirer in enumerate(rewirers):
        print("-------------------")
        print(f"Training a GIN model + interleaved {rewirer}...")
        graphs_train_rewirer = []
        for g in graphs_train:
            g.attach_rewirer(rewirer)
            graphs_train_rewirer.append(g.to_torch_data().to(device))

        graphs_val_rewirer = []
        for g in graphs_val:
            g.attach_rewirer(rewirer)
            graphs_val_rewirer.append(g.to_torch_data().to(device))

        dl_train = DataLoader(
            graphs_train_rewirer, batch_size=config.train.train_batch_size
        )
        dl_val = DataLoader(graphs_val_rewirer, batch_size=config.train.val_batch_size)

        in_channels = graphs_train_rewirer[0].x.shape[1]
        out_channels = (
            1
            if len(graphs_train_rewirer[0].y.shape) == 0
            else len(graphs_train_rewirer[0].y.shape)
        )

        model = GINModel(
            in_channels=in_channels,
            hidden_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers,
            out_channels=out_channels,
            drop_prob=config.model.drop_prob,
            interleave_diff_graph=True,
            global_pool_aggr=config.model.global_pool_aggr,
            norm=config.model.norm,
        )
        model.to(device)

        end_results = train_eval_loop(
            model,
            dl_train,
            dl_val,
            lr=config.train.lr,
            num_epochs=config.train.num_epochs,
            print_every=config.train.print_every,
            verbose=not config.run.silent,
            log_wandb=False,
            loss_fn=config.train.loss_fn,
            gamma_decay=config.train.gamma_decay,
            warmup=config.train.warmup,
            train_std=train_std,
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def run_experiment(config, graphs_train, graphs_val):
    print(config)
    os.environ["WANDB_SILENT"] = str(config.run.silent).lower()
    print(f"Device: {device}")

    assert (
        (config.model.approach == "only_original")
        or (config.model.approach == "only_diff")
        or (config.model.approach == "interleave")
    )

    if config.model.approach == "only_original":
        print("No rewiring.")
        rewirers = [None]
    elif config.model.approach == "only_diff":
        print("Only diff graph.")
        rewirers = ['fully_connected']
    else:
        rewirers = config.data.rewirers

    results = {rewirer: [] for rewirer in rewirers}

    train_mean = np.mean([g.y.cpu() for g in graphs_train])
    train_std = np.std([g.y.cpu() for g in graphs_train])
    print(f"train targets: {train_mean:.2f} +/- {train_std:.3f}")

    val_mean = np.mean([g.y.cpu() for g in graphs_val])
    val_std = np.std([g.y.cpu() for g in graphs_val])
    print(f"val targets: {val_mean:.2f} +/- {val_std:.3f}")

    in_channels = graphs_train[0].x.shape[1]
    out_channels = (
        1 if len(graphs_train[0].y.shape) == 0 else len(graphs_train[0].y.shape)
    )

    # run experiment for each rewirer

    for rewirer in rewirers:
        set_seed(config.data.seed)

        graphs_train_rewirer = []
        for g in graphs_train:
            if rewirer is not None:
                g.attach_rewirer(rewirer)
            graphs_train_rewirer.append(g.to_torch_data().to(device))

        graphs_val_rewirer = []
        for g in graphs_val:
            if rewirer is not None:
                g.attach_rewirer(rewirer)
            graphs_val_rewirer.append(g.to_torch_data().to(device))

        for seed in config.model.seeds:
            config.model.seed = seed

            set_seed(config.model.seed)

            print(f"Rewirer {rewirer} with seed {config.model.seed}")

            dl_train = DataLoader(
                graphs_train_rewirer, batch_size=config.train.train_batch_size
            )
            dl_val = DataLoader(
                graphs_val_rewirer, batch_size=config.train.val_batch_size
            )

            if config.data.name == "SalientDists":
                wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    config=config,
                    group=f"{config.wandb.experiment_name}-{config.model.approach}-{rewirer}-c1-{config.data.c1}-c2-{config.data.c2}-c3-{config.data.c3}",
                )
                wandb.run.name = f"{config.wandb.experiment_name}-{config.model.approach}-{rewirer}-c1-{config.data.c1}-c2-{config.data.c2}-c3-{config.data.c3}-seed-{config.model.seed}"
            elif config.data.name == "ColourInteract":
                wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    config=config,
                    group=f"{config.data.dataset}-{config.model.approach}-rewired-with-{rewirer}-c2/c1-{config.data.c2 / config.data.c1}",
                )
                wandb.run.name = f"{config.data.dataset}-{config.model.approach}-rewired-with-{rewirer}-c2/c1-{config.data.c2 / config.data.c1}-seed-{config.model.seed}"
            else:
                raise Exception

            wandb.log({"train/target_mean": train_mean, "train/target_std": train_std})
            wandb.log({"eval/target_mean": val_mean, "eval/target_std": val_std})

            model = GINModel(
                in_channels=in_channels,
                hidden_channels=config.model.hidden_channels,
                num_layers=config.model.num_layers,
                out_channels=out_channels,
                drop_prob=config.model.drop_prob,
                only_original_graph=(config.model.approach == "only_original"),
                interleave_diff_graph=(config.model.approach == "interleave"),
                only_diff_graph=(config.model.approach == "only_diff"),
                norm=config.model.norm,
            ).to(device)

            print(model)
            print(
                f"Total number parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
            )

            final_val_loss = train_eval_loop(
                model,
                dl_train,
                dl_val,
                lr=config.train.lr,
                num_epochs=config.train.num_epochs,
                print_every=config.train.print_every,
                verbose=not config.run.silent,
                log_wandb=True,
                loss_fn=config.train.loss_fn,
                gamma_decay=config.train.gamma_decay,
                warmup=config.train.warmup,
                train_std=train_std,
            )

            results[rewirer].append(final_val_loss)

            del model

            wandb.finish()

    with open("data/results/" + config.wandb.experiment_name + ".json", "w") as f:
        json.dump(results, f)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fn",
        default="debug_ColourInteract.yaml",
        help="configuration file name",
        type=str,
    )

    # for SalientDists
    parser.add_argument("--c1", default=1.0, help="c1", type=float)
    parser.add_argument("--c2", default=1.0, help="c2", type=float)
    parser.add_argument("--c3", default=1.0, help="c3", type=float)

    # for ColourInteract
    parser.add_argument("--c2_over_c1", default=1.0, help="c2/c1", type=float)

    args = parser.parse_args()

    with open(args.config_fn, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    print(config)

    results = {}
    
    if config.data.name == "SalientDists":
        
        config.data.c1 = args.c1
        config.data.c2 = args.c2
        config.data.c3 = args.c3

        print(config)

        graphs_train, graphs_val = get_data_SalientDists(
            dataset=config.data.dataset,
            c1=config.data.c1,
            c2=config.data.c2,
            c3=config.data.c3,
            d=config.data.d,
            normalise=config.data.normalise,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            min_val_nodes=config.data.min_val_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            verbose=config.run.silent,
        )

        for approach in config.model.approaches:
            config.model.approach = approach

            results = results | run_experiment(config, graphs_train, graphs_val)

    elif config.data.name == "ColourInteract":

        config.data.c1 = 1 / (1 + args.c2_over_c1)
        config.data.c2 = args.c2_over_c1 / (1 + args.c2_over_c1)

        assert math.isclose(
                    config.data.c1 + config.data.c2, 1.0
                )

        graphs_train, graphs_val = get_data_ColourInteract(
            dataset=config.data.dataset,
            c1=config.data.c1,
            c2=config.data.c2,
            normalise=config.data.normalise,
            num_colours=config.data.num_colours,
            min_train_nodes=config.data.min_train_nodes,
            max_train_nodes=config.data.max_train_nodes,
            min_val_nodes=config.data.min_val_nodes,
            max_val_nodes=config.data.max_val_nodes,
            train_size=config.data.train_size,
            val_size=config.data.val_size,
            verbose=config.run.silent,
        )
        for approach in config.model.approaches:
            config.model.approach = approach

            results = results | run_experiment(config, graphs_train, graphs_val)
    else:
        raise Exception

    pprint(results)

    # quick_run(
    #     [
    #         "cayley",
    #         # "fully_connected",
    #         "aligned_cayley",
    #         # "interacting_pairs"
    #     ],
    #   "debug_SalientDists_ZINC.yaml"
    #   )

    # quick_run(
    #     [
    #         # "fully_connected",
    #         "cayley_clusters",
    #         "cayley",
    #     ],
    #     "debug_ColourInteract.yaml"
    # )


if __name__ == "__main__":
    main()
