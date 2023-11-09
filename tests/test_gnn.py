import pytest
from openimpact.model.train import train_gnn
from pickle import load

import lightning as L  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from openimpact.data.datasets import train_test_split
from openimpact.model.gnn import FarmGNN


@pytest.fixture(scope="class")
def gnn_params():
    return {
        "lr": 0.002443162103996048,
        "weight_decay": 1e-5,
        "dim_inner": 312,
        "num_layers": 4,
        "att_heads": 1,
        "h_dim": 22,
    }


class TestGNN:
    def test_model_init(self, gnn_params):
        model = FarmGNN(3, 1, **gnn_params)
        assert model is not None

    def test_training_step(self, gnn_params, kelmarsh_dataset):
        train_data, val_data = train_test_split(kelmarsh_dataset)

        assert train_data is not None
        assert val_data is not None

        batch_size = 1

        train_loader = DataLoader(
            train_data, batch_size=batch_size, num_workers=1
        )
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=1)

        assert train_loader is not None
        assert val_loader is not None

        dim_in = train_data[0].num_node_features
        dim_out = (
            1 if len(train_data[0].y.shape) == 1 else train_data[0].y.shape[1]
        )

        with open("tests/scaler/GNN.pkl", "rb") as f:
            scaler = load(f)

        model = FarmGNN(
            dim_in=dim_in, dim_out=dim_out, scaler=scaler, **gnn_params
        )

        max_epochs = 1
        trainer = L.Trainer(max_epochs=max_epochs, logger=False)
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    def test_train_gnn(
        self, kelmarsh_dataset, gnn_params, log_dir, scaler_path
    ):
        train_params = {
            "batch_size": 1,
            "max_epochs": 1,
        }

        train_gnn(
            dataset=kelmarsh_dataset,
            gnn_params=gnn_params,
            train_params=train_params,
            scaler_path=scaler_path,
            experiment="GNN_test",
            log_dir=log_dir,
        )
