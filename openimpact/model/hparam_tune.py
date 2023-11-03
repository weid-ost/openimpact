from pickle import dump
import tomli_w

import lightning as L  # type: ignore
from farmgnn.loggers import DictLogger
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from farmgnn.datasets import load_dataset, train_test_split
from farmgnn.gnn import FarmGNN

from skopt.space import Real, Integer
from skopt import gp_minimize
from skopt import dump as skopt_dump
from skopt.callbacks import CheckpointSaver

import functools

import numpy as np
from pathlib import Path

np.int = np.int32

experiment = 5


def main():
    dataset = load_dataset()
    train_data, val_data = train_test_split(dataset)

    scaler = StandardScaler()
    scaler.fit(train_data.x.numpy())

    scaler_path = Path("scaler")
    scaler_path.mkdir(parents=True, exist_ok=True)
    dump(scaler, open(scaler_path / "StandardScaler.pkl", "wb"))

    dim_in = train_data[0].num_node_features
    dim_out = (
        1 if len(train_data[0].y.shape) == 1 else train_data[0].y.shape[1]
    )

    space = [
        # Integer(50, 1000, name="max_epochs"),
        Integer(1, 2, name="max_epochs"),
        Real(10**-5, 10**0, "log-uniform", name="lr"),
        Real(10**-5, 10**0, "log-uniform", name="weight_decay"),
        Integer(2, 512, name="dim_inner"),
        Integer(1, 10, name="num_layers"),
        Integer(1, 8, name="att_heads"),
        Integer(1, 32, name="h_dim"),
        Integer(1, 512, name="batch_size"),
    ]

    obj = functools.partial(
        objective,
        train_data=train_data,
        val_data=val_data,
        scaler=scaler,
        dim_in=dim_in,
        dim_out=dim_out,
    )

    checkpoint_saver = CheckpointSaver(
        f"hparam_tune/experiment_{experiment}/checkpoint.pkl",
        compress=9,
        store_objective=False,
    )

    res_gp = gp_minimize(obj, space, n_calls=10, callback=[checkpoint_saver])
    save_optimizer(res_gp, experiment)


def save_optimizer(res_gp, experiment):
    save_dict = {
        "fun": float(res_gp.fun),
        "x": np.array(res_gp.x).tolist(),
        "func_vals": np.array(res_gp.func_vals).tolist(),
        "x_iters": np.array(res_gp.x_iters).tolist(),
    }

    with open(f"hparam_tune/experiment_{experiment}/res_gp.toml", "wb") as fp:
        tomli_w.dump(save_dict, fp)

    skopt_dump(
        res_gp,
        f"hparam_tune/experiment_{experiment}/res_gp.pkl",
        store_objective=False,
    )


def objective(space, train_data, val_data, scaler, dim_in, dim_out):
    hparams = {
        "lr": space[1],
        "weight_decay": space[2],
        "dim_inner": space[3],
        "num_layers": space[4],
        "att_heads": space[5],
        "h_dim": space[6],
    }

    hparams_2 = {"max_epochs": int(space[0]), "batch_size": int(space[7])}

    train_loader = DataLoader(train_data, batch_size=hparams_2["batch_size"])
    val_loader = DataLoader(val_data, batch_size=hparams_2["batch_size"])
    model = FarmGNN(dim_in=dim_in, dim_out=dim_out, scaler=scaler, **hparams)

    loggers = [
        TensorBoardLogger("~/Projects/farmgnn"),
        DictLogger("~/Projects/farmgnn", experiment=experiment),
    ]

    loggers[1].log_hyperparams({**hparams, **hparams_2})

    trainer = L.Trainer(max_epochs=hparams_2["max_epochs"], logger=loggers)
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    return trainer.loggers[1].metrics["val_loss"][-1]


if __name__ == "__main__":
    main()
