import shutil
import pytest
import pandas as pd
import numpy as np
from openimpact.data.datasets import KelmarshDataset

np.random.seed(42)


@pytest.fixture(scope="module")
def raw_dataframe():
    index = pd.date_range(
        "2023-11-08 12:00:00",
        "2023-11-08 20:00:00",
        freq="10min",
        name="datetime",
    )

    n = len(index)

    data = {
        "Wind speed": np.random.rand(n) * 15.0,
        "Wind direction": np.random.rand(n) * 359.9,
        "Power": np.random.rand(n) * 1000.0,
        "Wind turbine ID": np.ones(n, dtype=int),
    }

    df = pd.DataFrame(data, index=index)

    nan_index = np.random.choice(index, 20)

    df.loc[nan_index, ["Wind speed", "Power"]] = np.nan

    df_wt_2 = df.copy()
    df_wt_2["Wind turbine ID"] = 2

    return pd.concat([df, df_wt_2]).copy(deep=True)


@pytest.fixture(scope="module")
def openimpact_dataframe(raw_dataframe):
    rename = {
        "Wind speed": "wind_speed",
        "Power": "power",
        "Wind turbine ID": "wt_id",
    }

    df = raw_dataframe.rename(columns=rename)[rename.values()]
    return df.copy(deep=True)


@pytest.fixture(scope="module")
def clean_dataframe(openimpact_dataframe):
    df = openimpact_dataframe.dropna(axis=0).query("power > 20.0")
    return df.copy(deep=True)


# TODO: Reduce the size of the dataset
@pytest.fixture(scope="module")
def kelmarsh_dataset():
    dataset = KelmarshDataset("tests/kelmarsh_test")

    idx = np.arange(len(dataset))
    sample_size = 10
    sample_idx = np.random.choice(idx, size=sample_size, replace=False)

    return dataset.copy(sample_idx)


@pytest.fixture
def scaler_path(tmp_path):
    scaler_path = tmp_path / "scaler"
    scaler_path.mkdir()
    return scaler_path


@pytest.fixture
def log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir
