# -*- coding: utf-8 -*-
"""
Functions for preprocessing

"""

__all__ = [
    "log_process",
    "join_frames",
    "add_time_cols",
    "square_root",
    "filter_constants",
    "filter_power",
    "resample",
    "filter_regexp",
    "dropna",
    "filter_corr",
    "compose",
    "ComposableFunction",
    "remove_interval",
]

import functools
from typing import Callable, Any

import numpy as np
import pandas as pd


def log_process(func):
    def wrapper(*args, **kwargs):
        df = args[0]
        set(df.columns)
        df.shape[0]
        result = func(*args, **kwargs)
        # removed_columns = columns - set(result.columns)
        # n_removed_rows = n_samples - result.shape[0]
        # logger.info(
        #     f"{func} Columns removed: {len(removed_columns)}, rows removed: {n_removed_rows} \n {removed_columns}"
        # )
        return result

    return wrapper


def join_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frame = frames[0].join(frames[1:], how="outer")
    return frame.loc[~frame.index.duplicated()]


def add_time_cols(df) -> pd.DataFrame:
    df["Year"] = df.index.year
    month_type = pd.CategoricalDtype(
        categories=[
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        ordered=True,
    )
    df["Month"] = df.index.month_name().astype(month_type)
    df["Day"] = df.index.day_name()
    df["Hour"] = df.index.hour
    df["Minute"] = df.index.minute
    return df


def square_root(x):
    return np.sqrt(np.square(x).sum())


def filter_constants(df) -> pd.DataFrame:
    df_tmp = df.std() / df.agg(square_root)

    tmp = df_tmp.loc[df_tmp < 1e-6].index.tolist()
    return df[df.columns[~df.columns.isin(tmp)]]


def filter_power(
    df, col: str, rated: float, thresh: float = 0.1
) -> pd.DataFrame:
    return df.loc[df[col] > rated * thresh]


def filter_bin(
    data,
    ws_col: str,
    p_col: str,
    sigma: float,
    cut_in: float,
    cut_out: float,
    bins=40,
) -> pd.DataFrame:
    windspeeds = np.linspace(cut_in, cut_out, bins)

    df = data.copy()

    df["bins"] = pd.cut(df[ws_col], bins=windspeeds)

    df = df.dropna(subset="bins")

    df_bin_mean = df.groupby("bins", observed=True).mean(numeric_only=True)
    df_bin_std = df.groupby("bins", observed=True).std(numeric_only=True)

    inlier_ids = []

    for row in df.iterrows():
        if (
            np.abs(row[1][p_col] - df_bin_mean.loc[row[1].bins][p_col])
            <= sigma * df_bin_std.loc[row[1].bins][p_col]
        ):
            inlier_ids.append(row[0])

    return df.drop("bins", axis=1).loc[inlier_ids]


def resample(df, period: str = "D") -> pd.DataFrame:
    return df.resample(period).mean()


def filter_regexp(df, regex: str = "^((?!Min).)*[^xn]$") -> pd.DataFrame:
    return df.filter(regex=regex)


def dropna(df, axis=1, thresh=1.0) -> pd.DataFrame:
    return df.dropna(axis=axis, thresh=int(thresh * df.shape[~axis & 1]))


def filter_corr(df, thresh: float = 0.95, pre_choice=None) -> pd.DataFrame:
    corr = df.corr()
    corr = corr - np.eye(corr.shape[0])
    mask = corr.abs() > 0.95
    to_filter = corr[mask].unstack().dropna().to_frame()

    keep = set(pre_choice) if pre_choice else set()
    discard = set()

    for i, j in to_filter.index:
        if i not in discard:
            keep.add(i)
        if j not in keep:
            discard.add(j)

    return df[df.columns[~df.columns.isin(discard)]]


ComposableFunction = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*functions: ComposableFunction) -> ComposableFunction:
    return functools.reduce(lambda f, g: lambda x: g(f(x)), functions)


def remove_interval(
    df: pd.DataFrame,
    timestamps: list[str | pd.Timestamp],
    delta: str | pd.Timedelta = "30D",
):
    mask = np.ones(len(df.index), dtype=bool)

    for timestamp in timestamps:
        start = pd.Timestamp(timestamp) - pd.Timedelta(delta)
        end = pd.Timestamp(timestamp) + pd.Timedelta(delta)

        mask = np.logical_and(mask, (df.index < start) | (df.index > end))

    return df.loc[mask]


def match_group_count(
    df: pd.DataFrame, by: str, count_col: str, match: int | float | str
) -> pd.DataFrame:
    grouped = df.groupby(by)
    return grouped.filter(lambda x: x[count_col].count() == match)


def rename_labels(s: pd.Series, mapping_dict: dict) -> pd.Series:
    return s.apply(lambda x: mapping_dict[x])
