# -*- coding: utf-8 -*-

"""
Various functions getting the distances and azimuth angles between two points etc.

# Usage

Get distance matrix for all turbines and met mast
dist_matrix = distance_matrix(
    df_positions, lat_col="Lat.", long_col="Long.", name_col="WTG"
)

df_azimuth = azimuth_matrix(
    df_positions, lat_col="Lat.", long_col="Long.", name_col="WTG"
)

In case a dict with location name and tuple(lat, long) is used

locations = {
    "Metmast": (-4.19, -38.08),  # (Lat., Lon.)
    "WTG1": (-4.18523, -38.08384),
    "WTG2": (-4.182794, -38.085799),
    "WTG6": (-4.185351, -38.087376),
    "WTG11": (-4.18908, -38.08262),
    "WTG12": (-4.189967, -38.082009),
    "WTG13": (-4.190953, -38.081391),
    "WTG14": (-4.191046, -38.079538),
    "WTG15": (-4.19211, -38.07853),
    "WTG17": (-4.19395, -38.07708),
    "WTG20": (-4.19679, -38.07462),
    "WTG26": (-4.20256, -38.07288),
    "WTG30": (-4.20724, -38.06941),
    "WTG25": (-4.20965, -38.06663),
}

dist_matrix = distance_matrix_dict(locations)

"""

import itertools

import haversine
import numpy as np
import pandas as pd
import pyproj


def distance_matrix_from_df(
    df: pd.DataFrame, lat_col: str, long_col: str, name_col: str | None = None
) -> pd.DataFrame:
    lat_long = tuple(zip(df[lat_col], df[long_col]))
    dim = df.shape[0]

    distance_matrix = np.array(
        [
            haversine.haversine(i, j)
            for i, j in itertools.product(lat_long, repeat=2)
        ]
    ).reshape(dim, dim)

    columns = df[name_col] if name_col else pd.RangeIndex(0, dim)

    return pd.DataFrame(data=distance_matrix, index=columns, columns=columns)


def distance_matrix_from_dict(
    locations: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    dim = len(locations)

    distance_matrix = np.array(
        [
            haversine.haversine(i[1], j[1])
            for i, j in itertools.product(locations.items(), repeat=2)
        ]
    ).reshape(dim, dim)

    columns = list(locations.keys())

    return pd.DataFrame(data=distance_matrix, index=columns, columns=columns)


def azimuth_matrix_from_df(
    df: pd.DataFrame, lat_col: str, long_col: str, name_col: str | None = None
) -> pd.DataFrame:
    long_lat = tuple(zip(df[long_col], df[lat_col]))
    dim = df.shape[0]

    geodesic = pyproj.Geod(ellps="WGS84")
    azimuth_matrix = np.array(
        [
            geodesic.inv(*i, *j)[0]
            for i, j in itertools.product(long_lat, repeat=2)
        ]
    ).reshape(dim, dim)

    columns = df[name_col] if name_col else pd.RangeIndex(0, dim)

    return pd.DataFrame(data=azimuth_matrix, index=columns, columns=columns)


def azimuth_matrix_from_dict(
    locations: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    dim = len(locations)

    # Important: Need to change order from (lat, long) to (long, lat)
    # TODO: Find more elagant way
    locations = {k: (v[1], v[0]) for k, v in locations.items()}

    geodesic = pyproj.Geod(ellps="WGS84")
    azimuth_matrix = np.array(
        [
            geodesic.inv(*i, *j)[0]
            for i, j in itertools.product(locations.values(), repeat=2)
        ]
    ).reshape(dim, dim)

    columns = list(locations.keys())

    return (
        pd.DataFrame(data=azimuth_matrix, index=columns, columns=columns)
        - np.eye(dim) * 180.0
    ) % 360.0


def lat_lon_to_xy(locations: dict[str, tuple[float, float]]) -> list[tuple]:
    P = pyproj.Proj("WGS84", preserve_units=False)

    lats = [v[0] for v in locations.values()]
    longs = [v[1] for v in locations.values()]

    x, y = P(longs, lats)

    x = x / np.min(x) - 1.0
    y = y / np.min(y) - 1.0

    return list(zip(locations.keys(), x, y))


def flatten(list_of_lists: list[list]) -> list:
    return [item for sublist in list_of_lists for item in sublist]


def stack(li: list) -> list[list]:
    return [[item] for item in li]
