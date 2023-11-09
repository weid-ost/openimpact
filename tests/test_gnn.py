import pytest
from openimpact.model.train import train_gnn
from pickle import dump, load

import lightning as L  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from lightning.pytorch.loggers import CSVLogger

from openimpact.data.datasets import load_dataset, train_test_split
from openimpact.model.gnn import FarmGNN


@pytest.fixture
def gnn_params():
    return {
        "lr": 0.002443162103996048,
        "weight_decay": 1e-5,
        "dim_inner": 312,
        "num_layers": 4,
        "att_heads": 1,
        "h_dim": 22,
    }


def test_model_init(gnn_params):
    model = FarmGNN(3, 1, **gnn_params)
    assert model is not None


# def test_model_forward():
#     model = FarmGNN(2, 1, 1, 1, 1)
#     input_data = torch.randn((1, 2))
#     output = model(input_data)
#     assert output is not None


# def test_lightning_module_training_step():
#     # Test the training step
#     model = MyLightningModule()
#     batch = next(iter(dataloader))  # Replace with your dataloader
#     loss = model.training_step(batch, 0)
#     assert loss is not None
#     assert 'loss' in loss
#
# Test training step
def test_training_step(gnn_params, kelmarsh_dataset):
    train_data, val_data = train_test_split(kelmarsh_dataset)

    scaler = StandardScaler()
    scaler.fit(train_data.x.numpy())

    batch_size = 1

    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

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
    trainer = L.Trainer(max_epochs=max_epochs, logger=None)
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
