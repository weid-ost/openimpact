from pathlib import Path

import pandas as pd

from openimpact.data.data_store import kelmarsh_raw_data
from openimpact.data.wranglers import (
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
)
from openimpact.model.features import add_freestream_conditions


def kelmarsh_pipeline():
    """The goal of the pipeline is to have several tasks / steps are (1) replacable / changable,
    (2) scalable and (3) skippable.

    In case the problem changes, each individual step should be able to be replaced, if needed.
    In case the problem becomes bigger in terms of data and hence reuqired processing power,
    each individual step should be able to be scaled as necessary.
    In case the pipeline is run several times, e.g. a few steps have been replaced or changed,
    only the minimum amount of required steps should be executed again. If the raw data has been downloaded and wrangled already,
    then they should be automatically run again.
    """
    # Load raw data and save in a DataFrame
    download_path = (
        Path("~/Projects/openimpact-lib/data/").expanduser().absolute()
    )
    file_paths = kelmarsh_raw_data(download_path)

    # Data wrangling
    default_cfg_path = (
        Path(
            "~/Projects/openimpact-lib/openimpact/data/datasets/kelmarsh.toml"
        )
        .expanduser()
        .absolute()
    )
    set_default_cfg(default_cfg_path)

    raw_data_path = file_paths[0]
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

    df.to_csv(download_path / "cleaned.csv")

    # Feature engineering
    # Here we need to make sure to create the features that are needed for the PyTorch Geometric Dataset
    df = add_freestream_conditions(df)
    df.to_csv(download_path / "features.csv")

    # Next we need to create a PyTorch Geometric Dataset


def main():
    kelmarsh_pipeline()


if __name__ == "__main__":
    main()
