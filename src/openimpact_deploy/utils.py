# -*- coding: utf-8 -*-
"""Utility functions"""

from pathlib import Path
import pickle
from typing import Union, Any
import pandas as pd
import yaml


def read_config(file_path):
    with open(file_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    return config


def save_data(df: pd.DataFrame, filename: Union[str, Path]):
    df.to_pickle(f"{filename}.pkl")


def load_data(filename: Union[str, Path]) -> pd.DataFrame:
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
