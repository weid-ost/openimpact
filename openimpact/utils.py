# -*- coding: utf-8 -*-
"""Functions for describing wind farm layouts and creating wind farm representations"""

import os
from pathlib import Path
import pickle
from typing import Union, Any, TypeVar
import pandas as pd
import tomllib

T = TypeVar("T")


def read_config(file_path: str | Path) -> dict:
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    return config


def save_data(df: pd.DataFrame, filename: Union[str, Path]):
    df.to_pickle(f"{filename}.pkl")


def load_data(filename: Union[str, Path]) -> pd.DataFrame | pd.Series:
    return pd.read_pickle(f"{filename}.pkl")


def date_to_str(t) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def pickle_object(obj: Any, filename: Union[str, Path]):
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: Union[str, Path]) -> Any:
    with open(f"{filename}", "rb") as f:
        obj = pickle.load(f)
    return obj


def getenv(key: str, default: T = 0) -> T:
    return type(default)(os.getenv(key, default))
