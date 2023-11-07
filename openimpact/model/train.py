from pickle import dump

import lightning as L  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from lightning.pytorch.loggers import CSVLogger

from openimpact.data.datasets import load_dataset, train_test_split
from openimpact.model.gnn import FarmGNN


def train_gnn(
    dataset_dir: str,
    gnn_params: dict,
    train_params: dict = {},
    scaler_path: str = "scaler",
    experiment: str = "GNN",
):
    dataset = load_dataset(dataset_dir)
    train_data, val_data = train_test_split(dataset)

    scaler = StandardScaler()
    scaler.fit(train_data.x.numpy())

    dump(scaler, open(f"{scaler_path}/{experiment}.pkl", "wb"))

    batch_size = train_params.get("batch_size", 1)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    dim_in = train_data[0].num_node_features
    dim_out = (
        1 if len(train_data[0].y.shape) == 1 else train_data[0].y.shape[1]
    )

    model = FarmGNN(
        dim_in=dim_in, dim_out=dim_out, scaler=scaler, **gnn_params
    )

    logger = CSVLogger("logs", f"{experiment}")
    max_epochs = train_params.get("max_epochs", 200)
    trainer = L.Trainer(max_epochs=max_epochs, logger=logger)
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
