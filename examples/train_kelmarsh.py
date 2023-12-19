from pathlib import Path

from openimpact.data.wranglers import (
    get_dataset,
    set_default_cfg,
)
from openimpact.data.datasets.kelmarsh import KelmarshDataset

from openimpact.model.train import train_gnn
from openimpact.utils import getenv


if __name__ == "__main__":
    default_cfg_path = Path(getenv("CONFIG_PATH", "examples/kelmarsh.toml"))
    set_default_cfg(default_cfg_path)

    gnn_params = {
        "lr": 0.002443162103996048,
        "weight_decay": 1e-5,
        "dim_inner": 312,
        "num_layers": 4,
        "att_heads": 1,
        "h_dim": 22,
    }
    train_params = {
        "batch_size": 512,
        "max_epochs": 2,
    }

    dataset = get_dataset()
    kelmarsh_data = KelmarshDataset(
        dataset["root"],
        features=["u_g", "v_g", "nacelle_direction"],
        target="wind_speed",
        wt_col="wt_id",
    )

    train_gnn(
        dataset=kelmarsh_data,
        gnn_params=gnn_params,
        train_params=train_params,
    )
