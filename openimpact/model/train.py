from pickle import dump
from pathlib import Path

import lightning as L  # type: ignore
from torch_geometric.data import InMemoryDataset  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from lightning.pytorch.loggers import CSVLogger

from openimpact.data.datasets import train_test_split
from openimpact.model.gnn import FarmGNN, FarmGNN3, FarmGAT


def train_gnn(
    dataset: InMemoryDataset,
    gnn_params: dict,
    train_params: dict = {},
    scaler_path: str | Path = Path("scaler"),
    log_dir: str | Path = Path("logs"),
    experiment: str = "GNN",
):
    """
    currently testing this combination:

    h_dims = [256, 256, 256]
    heads = [8, 8, 8]
    concats = [True, True, True]
    edge_dims = [2, None, None]
    skip_connections = [True, True, True]
    """
    train_data, val_data = train_test_split(dataset)

    batch_size = train_params.get("batch_size", 1)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    dim_in = train_data[0].num_node_features
    dim_out = (
        1 if len(train_data[0].y.shape) == 1 else train_data[0].y.shape[1]
    )

    model = FarmGAT(
        dim_in,
        dim_out,
        **gnn_params,
    )

    logger = CSVLogger(log_dir, f"{experiment}")
    max_epochs = train_params.get("max_epochs", 200)
    trainer = L.Trainer(
        max_epochs=max_epochs, logger=logger, accelerator="gpu"
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
