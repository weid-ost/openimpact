import functools
import scipy
import pandas as pd
import numpy as np


def add_freestream_conditions(df: pd.DataFrame) -> pd.DataFrame:
    circ_mean = functools.partial(scipy.stats.circmean, high=360)

    df["ws_g"] = df.groupby("datetime").aggregate({"wind_speed": "max"})
    df["wd_g"] = df.groupby("datetime").aggregate(
        {"wind_direction": circ_mean}
    )

    df["u_g"] = -np.cos(df["wd_g"] / 180.0 * np.pi) * df["ws_g"]
    df["v_g"] = -np.sin(df["wd_g"] / 180.0 * np.pi) * df["ws_g"]

    return df.copy()
