from pathlib import Path
import lightning as L
import numpy as np
from numpy._typing import NDArray
import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, GATv2Conv

__all__ = [
    "BaseGNN",
    "FarmGNN",
    "FarmGNN3",
    "FarmGAT",
    "get_checkpoint",
    "load_model",
]


class BaseGNN(L.LightningModule):
    """
    Base GNN
    """

    def __init__(self, lr, weight_decay):
        super().__init__()

        self.lr = lr
        self.weight_decay

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        y_hat = self((x, edge_index, edge_attr))
        loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        y_hat = self((x, edge_index, edge_attr))
        val_loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("val_loss", val_loss, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        y_hat = self((x, edge_index, edge_attr))
        test_loss = F.mse_loss(y_hat, batch.y.reshape(batch.num_nodes, -1))

        self.log("test_loss", test_loss)
        return test_loss

    def predict_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        return self((x, edge_index, edge_attr))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, x, edge_index, edge_attr):
        raise NotImplementedError()


class FarmGNN(BaseGNN):
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

    Parameters
    ----------
    dim_in:
        Input dimension
    dim_out:
        Output dimension
    **kwargs:
        Optional additional args
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
        super().__init__(lr, weight_decay)

        self.dim_in = dim_in

        self.dim_out = dim_out

        self.dim_inner = dim_inner
        self.num_layers = num_layers

        self.att_heads = att_heads

        self.h_dim = h_dim

        self._A = []

        self.scaler = scaler

        self.n_nodes = None

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

    @property
    def A(self):
        return self._A

    @property
    def att_matrix(self) -> NDArray:
        att_matt = np.ones((self.n_nodes, self.n_nodes)) * -1
        for edge, (i, j) in enumerate(self.att_edges):
            att_matt[i, j] = self._A[1][edge, 0]
        return att_matt

    @property
    def att_edges(self) -> list:
        return list(zip(self._A[0][0].numpy(), self._A[0][1].numpy()))


class FarmGNN3(BaseGNN):
    """
    Windfarm GNN model: encoder + stage + head


    Parameters
    ----------
    dim_in:
        Input dimension
    dim_out:
        Output dimension
    **kwargs:
        Optional additional args
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        lr: float = 0.005,
        weight_decay: float = 5e-4,
        optimizer: str = "Adam",
        scaler=None,
    ):
        super().__init__(lr, weight_decay)

        self.dim_in = dim_in

        self.dim_out = dim_out

        self.scaler = scaler

        self.n_nodes = None

        self.gat_1 = GATv2Conv(
            in_channels=self.dim_in,
            out_channels=256,
            heads=4,
            concat=True,
            edge_dim=2,
        )

        self.gat_post_1 = torch.nn.Sequential(torch.nn.LeakyReLU())

        self.skip_con1 = torch.nn.Linear(self.dim_in, 256 * 4, bias=False)

        self.gat_2 = GATv2Conv(
            in_channels=4 * 256,
            out_channels=256,
            heads=4,
            concat=True,
            edge_dim=2,
        )

        self.gat_post_2 = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
        )

        self.skip_con2 = torch.nn.Linear(256 * 4, 256 * 4, bias=False)

        self.gat_3 = GATv2Conv(
            in_channels=4 * 256,
            out_channels=256,
            heads=4,
            concat=True,
            edge_dim=2,
        )

        self.gat_post_3 = torch.nn.Sequential(torch.nn.LeakyReLU())

        self.skip_con3 = torch.nn.Linear(256 * 4, 256 * 4, bias=False)

        self.mp = torch.nn.Sequential(
            MLP(
                in_channels=4 * 256,
                hidden_channels=512,
                num_layers=8,
                out_channels=self.dim_out,
            ),
        )

        self.save_hyperparameters()

    def forward(self, x, edge_index, edge_attr):
        """ """
        # Will be used to return the V x V attention matrix, where V is the number of nodes
        self.n_nodes = x.shape[0]

        if self.scaler:
            x = torch.from_numpy(self.scaler.transform(x.numpy()))

        input = x
        x = self.gat_1(x, edge_index, edge_attr)
        x = self.gat_post_1(x)
        x += self.skip_con1(input)

        input = x
        x = self.gat_2(x, edge_index, edge_attr)
        x = self.gat_post_2(x)
        x += self.skip_con2(input)

        input = x
        x = self.gat_3(x, edge_index, edge_attr)
        x = self.gat_post_3(x)
        x += self.skip_con3(input)

        x = self.mp(x)

        return x


class FarmGAT(BaseGNN):
    """
    Windfarm GNN model: encoder + stage + head

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        h_dims,
        heads,
        concats,
        edge_dims,
        skip_connections,
        n_mlp_channels,
        n_mlp_layers,
        lr: float = 0.005,
        weight_decay: float = 5e-4,
        scaler=None,
    ):
        self.lr = lr
        self.weight_decay = weight_decay

        super().__init__(lr, weight_decay)

        self.scaler = scaler

        self.n_nodes = None

        gat_layer_params = [
            h_dims,
            heads,
            concats,
            edge_dims,
            skip_connections,
        ]

        if len(set(map(len, gat_layer_params))) not in (0, 1):
            raise ValueError(
                "The parameter lists for the GAT layers need to have the same length"
            )

        gat_layers = []
        in_dim = in_channels
        for (
            h_dim,
            n_heads,
            concat,
            edge_dim,
            skip_con,
        ) in zip(*gat_layer_params):
            gat_layers.append(
                GATLayer(in_dim, h_dim, n_heads, concat, edge_dim, skip_con)
            )
            if concat:
                in_dim = h_dim * n_heads
            else:
                in_dim = h_dim

        self.gat = torch.nn.Sequential(*gat_layers)

        if concats[-1]:
            final_layer_in = heads[-1] * h_dims[-1]
        else:
            final_layer_in = h_dims[-1]

        self.mp = torch.nn.Sequential(
            MLP(
                in_channels=final_layer_in,
                hidden_channels=n_mlp_channels,
                num_layers=n_mlp_layers,
                out_channels=out_channels,
            ),
        )

        self.save_hyperparameters()

    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.scaler:
            x = torch.from_numpy(self.scaler.transform(x.numpy()))

        x, edge_index, edge_attr = self.gat((x, edge_index, edge_attr))

        x = self.mp(x)

        return x


class GATLayer(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        concat=True,
        edge_dim=None,
        skip_connection=True,
    ):
        super().__init__()

        self.gat_conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim,
        )
        self.edge_dim = edge_dim

        self.gat_post = F.leaky_relu

        self.skip_con = None

        if skip_connection:
            if concat:
                self.skip_con = torch.nn.Linear(
                    in_channels, out_channels * heads, bias=False
                )
            else:
                self.skip_con = torch.nn.Linear(
                    in_channels, out_channels, bias=False
                )

    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.edge_dim is None:
            edge_attr = None
        input = x
        x = self.gat_conv(x, edge_index, edge_attr)
        x = self.gat_post(x)
        if self.skip_con is not None:
            x += self.skip_con(input)
        return x, edge_index, edge_attr


def load_model(checkpoint, gnn):
    model = gnn.load_from_checkpoint(checkpoint)

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


if __name__ == "__main__":
    in_channels = 3
    out_channels = 1
    h_dims = [256, 256, 256]
    heads = [4, 4, 8]
    concats = [True, True, True]
    edge_dims = [2, None, None]
    skip_connections = [True, True, True]
    gat_gnn = FarmGAT(
        in_channels,
        out_channels,
        h_dims,
        heads,
        concats,
        edge_dims,
        skip_connections,
        512,
        1,
    )
    print(gat_gnn)
