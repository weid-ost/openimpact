from pathlib import Path

import pandas as pd

from openimpact.data.data_store import kelmarsh_raw_data
from openimpact.data.wranglers import (
    get_dataset,
    set_default_cfg,
    to_openimpact,
    read_raw,
    get_column_mapping,
)
from openimpact.data.preprocessing import (
    dropna,
    filter_bin,
    filter_power,
    match_group_count,
    rename_labels,
)
from openimpact.model.features import add_freestream_conditions
from openimpact.data.datasets.kelmarsh import KelmarshDataset

from openimpact.model.train import train_gnn
from openimpact.utils import getenv


def data_pipeline():
    """The goal of the pipeline is to have several tasks / steps are (1) replacable / changable,
    (2) scalable and (3) skippable.

    In case the problem changes, each individual step should be able to be replaced, if needed.
    In case the problem becomes bigger in terms of data and hence reuqired processing power,
    each individual step should be able to be scaled as necessary.
    In case the pipeline is run several times, e.g. a few steps have been replaced or changed,
    only the minimum amount of required steps should be executed again. If the raw data has been downloaded and wrangled already,
    then they should be automatically run again.
    """

    dataset = get_dataset()

    download_path = Path(getenv("DOWNLOAD_PATH", "./data"))
    data_path = download_path / dataset["data"]

    if not data_path.exists():
        # Load raw data and save in a DataFrame
        kelmarsh_raw_data(download_path)

        # Data wrangling

        raw_data_path = download_path / "kelmarsh_raw.csv"
        df_raw = read_raw(raw_data_path)

        column_mapping = get_column_mapping()

        df = to_openimpact(df_raw, column_mapping=column_mapping)

        # Date cleaning and filtering
        df = dropna(df, axis=0)

        group_list = []

        for _, group in df.groupby("wt_id"):
            group = filter_power(group, "power", 2050, thresh=0.01)
            group = filter_bin(group, "wind_speed", "power", 2.0, 3.5, 22.0)
            group_list.append(group)

        df = pd.concat(group_list)

        df = match_group_count(df, by="datetime", count_col="wt_id", match=6)

        # Feature engineering
        # Here we need to make sure to create the features that are needed for the PyTorch Geometric Dataset
        df = add_freestream_conditions(df)

        wt_mapping = {
            228: "KWF1",
            229: "KWF2",
            230: "KWF3",
            231: "KWF4",
            232: "KWF5",
            233: "KWF6",
        }
        df["wt_id"] = rename_labels(df["wt_id"], wt_mapping)

        save_dataframe(df, data_path)

    windfarm_static_path = download_path / dataset["static"]
    # Next we need to create a PyTorch Geometric Dataset
    KelmarshDataset(
        dataset["name"],
        data_path=data_path,
        windfarm_static_path=windfarm_static_path,
        features=["u_g", "v_g", "nacelle_direction"],
        target="wind_speed",
        wt_col="wt_id",
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path)


if __name__ == "__main__":
    default_cfg_path = Path(getenv("CONFIG_PATH", "examples/kelmarsh.toml"))
    set_default_cfg(default_cfg_path)

    data_pipeline()
