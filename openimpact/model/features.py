import functools
from numpy.typing import NDArray
from pathlib import Path
from typing import Any
import scipy
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer


def get_freestream_conditions(df: pd.DataFrame) -> Any:
    circ_mean = functools.partial(scipy.stats.circmean, high=360)

    ws_g = df.groupby("datetime").aggregate({"wind_speed_norm": "max"})
    wd_g = df.groupby("datetime").aggregate({"wind_direction": circ_mean})

    u_g = np.cos(np.deg2rad(wd_g.to_numpy())) * ws_g.to_numpy()
    v_g = np.sin(np.deg2rad(wd_g.to_numpy())) * ws_g.to_numpy()

    X = np.column_stack([u_g.flatten(), v_g.flatten()])
    df_ = pd.DataFrame(X, columns=["u_g", "v_g"])

    df_["datetime"] = ws_g.index
    df_["u_g"] = df_["u_g"].astype(float)
    df_["v_g"] = df_["v_g"].astype(float)
    return df_


def normalise_wind_speed(x: pd.DataFrame | NDArray) -> NDArray:
    transformer = FunctionTransformer(np.log1p, validate=True)
    return transformer.fit_transform(x)

    # TODO: Properly implement the following. This will allow
    # to save the scaler and use it on new data later on.
    # Path(scaler_path).mkdir(exist_ok=True, parents=True)
    # with open(f"{scaler_path}/{experiment}.pkl", "wb") as f:
    # dump(scaler, f)


def normalise_angular_values(x: pd.DataFrame | NDArray) -> NDArray:
    """
    TODO: Could also use the polar coordinates of complex numbers to also
    account for the magnitude if desired.
    """
    u = np.cos(np.deg2rad(np.array(x)))
    v = np.sin(np.deg2rad(np.array(x)))
    return np.column_stack([u.flatten(), v.flatten()])
