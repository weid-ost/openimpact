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


@pytest.fixture
def kelmarsh_dataset():
    return KelmarshDataset("kelmarsh_test")