from pathlib import Path
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, GATv2Conv


class FarmGNN(L.LightningModule):
    """
    Windfarm GNN model: encoder + stage + head

    current_best_model = {
        "l_pre": 2,
        "l_mp": 2,
        "l_post": 2,
        "stage": "skipsum",
        "act": "prelu",
        "att_heads": 1,
        "dim_inner": 32,
        "batchnorm": True,
        "batch_size": 32,
    }

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """

    def __init__(
        self,
        dim_in: int | tuple[int, int],
        dim_out: int,
        dim_inner: int,
        num_layers: int,
        h_dim: int,
        att_heads: int = 1,
        lr: float = 0.005,
        weight_decay: float = 5e-4,
        optimizer: str = "Adam",
        scaler=None,
    ):
        super().__init__()

        self.dim_in = dim_in

        self.dim_out = dim_out

        self.dim_inner = dim_inner
        self.num_layers = num_layers

        self.att_heads = att_heads

        self.h_dim = h_dim

        self._A = None

        self.scaler = scaler

        self.n_nodes = None

        self.lr = lr
        self.weight_decay = weight_decay

        self.optimizer = optimizer

        self.gat = GATv2Conv(
            in_channels=self.dim_in,
            out_channels=self.h_dim,
            heads=self.att_heads,
            concat=False,
            edge_dim=2,
        )

        self.gat_post = torch.nn.Sequential(
            torch.nn.PReLU(),
        )

        self.mp = torch.nn.Sequential(
            MLP(
                in_channels=self.h_dim,
                hidden_channels=self.dim_inner,
                num_layers=self.num_layers,
                out_channels=self.dim_out,
            ),
        )

        self.save_hyperparameters()

    def forward(self, x, edge_index, edge_attr):
        # Will be used to return the V x V attention matrix, where V is the number of nodes
        self.n_nodes = x.shape[0]

        if self.scaler:
            x = torch.from_numpy(self.scaler.transform(x.numpy()))
        x, self._A = self.gat(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        x = self.gat_post(x)

        x = self.mp(x)

        return x

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        y_hat = self(x, edge_index, edge_attr)
        loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y_hat = self(x, edge_index, edge_attr)
        val_loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        y_hat = self(x, edge_index, edge_attr)
        test_loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("test_loss", test_loss)
        return test_loss

    def predict_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        return self(x, edge_index, edge_attr)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @property
    def A(self):
        return self._A

    @property
    def att_matrix(self):
        att_matt = np.ones((self.n_nodes, self.n_nodes)) * -1
        for edge, (i, j) in enumerate(self.att_edges):
            att_matt[i, j] = self._A[1][edge, 0]
        return att_matt

    @property
    def att_edges(self):
        return list(zip(self._A[0][0].numpy(), self._A[0][1].numpy()))


def load_model(checkpoint):
    model = FarmGNN.load_from_checkpoint(checkpoint)

    return model


def get_checkpoint(root_dir: str | Path, version: int | None = None):
    root_dir = Path(root_dir)
    if version is None or version == -1:
        versions = sorted(
            [
                int(path.name.split("_")[-1])
                for path in root_dir.glob("version_*")
            ]
        )
        version = versions[-1]

    checkpoint = list(
        (root_dir / f"version_{version}/checkpoints/").glob("*.ckpt")
    )[0]

    return checkpoint
