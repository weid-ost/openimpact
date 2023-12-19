# -*- coding: utf-8 -*-
"""Functions for describing wind farm layouts and creating wind farm representations"""

import copy
from typing import Any

import networkx as nx
import numpy as np

from openimpact.postpro import get_wind_direction

from .distance import (
    azimuth_matrix_from_dict,
    distance_matrix_from_dict,
    lat_lon_to_xy,
)


def create_graph(lat_lon_dict: dict, max_dist: float = 1e12) -> nx.Graph:
    df_dist = distance_matrix_from_dict(lat_lon_dict)
    df_azim = azimuth_matrix_from_dict(lat_lon_dict)

    df_x = df_dist * np.cos(df_azim / 180.0 * np.pi)
    df_y = df_dist * np.sin(df_azim / 180.0 * np.pi)

    dod = {}
    for i, j in df_dist.iterrows():
        node_dist = j.where((j < max_dist) & (i != j.index)).dropna()
        out_edges = node_dist.index

        node_x_dirs = df_x[i][out_edges]
        node_y_dirs = df_y[i][out_edges]
        dists = df_dist[i][out_edges]

        node_dicts = [
            {"x_dist": x, "y_dist": y, "dist": d}
            for x, y, d in zip(node_x_dirs, node_y_dirs, dists)
        ]
        dod[i] = dict(zip(out_edges, node_dicts))

    DG = nx.from_dict_of_dicts(dod, create_using=nx.DiGraph)

    positions = lat_lon_to_xy(lat_lon_dict)

    DG = update_positions(DG, positions)

    return DG


def update_positions(
    graph: nx.Graph, positions: list[tuple[Any, float, float]]
) -> nx.Graph:
    g = copy.deepcopy(graph)
    for name, x, y in positions:
        g.nodes[name]["pos"] = (x, y)
        g.nodes[name]["xi"] = x
        g.nodes[name]["yi"] = y
    return g


def update_states(
    graph: nx.Graph, state: str, states: list[tuple]
) -> nx.Graph:
    g = copy.deepcopy(graph)
    for name, *val in states:
        g.nodes[name][state] = tuple(val) if len(val) > 1 else val[0]
    return g


def filter_wd(
    graph: nx.Graph, tol: float = 15.0, wd_col: str = "wind_direction"
) -> nx.Graph:
    def filter_edge(n1, n2):
        x_dist, y_dist = graph[n1][n2].get("x_dist"), graph[n1][n2].get(
            "y_dist"
        )
        edge_dir = get_wind_direction(x_dist, y_dist)
        # wind_dir = graph.nodes[n1].get(wd_col)
        wind_dir = get_wind_direction(
            graph.nodes[n1].get("u"), graph.nodes[n1].get("v")
        )
        diff = edge_dir - wind_dir

        return abs((diff - 180) % 360 - 180) < tol

    view = nx.subgraph_view(graph, filter_edge=filter_edge)

    return nx.Graph(view.copy())
