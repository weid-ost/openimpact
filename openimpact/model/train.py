from pickle import dump

import lightning as L  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from lightning.pytorch.loggers import CSVLogger

from openimpact.data.datasets import load_dataset, train_test_split
from openimpact.model.gnn import FarmGNN


def train_gnn():
    dataset = load_dataset("dataset_test")
    train_data, val_data = train_test_split(dataset)

    scaler = StandardScaler()
    scaler.fit(train_data.x.numpy())

    dump(scaler, open("scaler/StandardScalerKelmarsh.pkl", "wb"))

    train_loader = DataLoader(train_data, batch_size=512)
    val_loader = DataLoader(val_data, batch_size=512)

    dim_in = train_data[0].num_node_features
    dim_out = (
        1 if len(train_data[0].y.shape) == 1 else train_data[0].y.shape[1]
    )

    hparams = {
        "lr": 0.002443162103996048,
        "weight_decay": 1e-5,
        "dim_inner": 312,
        "num_layers": 4,
        "att_heads": 1,
        "h_dim": 22,
    }

    model = FarmGNN(dim_in=dim_in, dim_out=dim_out, scaler=scaler, **hparams)

    logger = CSVLogger("logs", "test_logs")
    trainer = L.Trainer(max_epochs=2, logger=logger)
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
