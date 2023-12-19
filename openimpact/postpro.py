from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch_geometric import utils


def plot_power_curves(
    model, dataset, meas: bool = False, prefix: str | None = None
):
    turbines = range(dataset[0].num_nodes)
    with torch.no_grad():
        for turbine in turbines:
            y_pred = []
            x = []
            y = []

            for data in dataset:
                y_pred.append(
                    model(data.x, data.edge_index, data.edge_attr)
                    .detach()
                    .numpy()[turbine]
                )

                if meas:
                    y.append(data.y[turbine].numpy())

                u = data.x.numpy()[turbine, 0]
                v = data.x.numpy()[turbine, 1]
                wind_speed = np.sqrt(np.square(u) + np.square(v))

                x.append(wind_speed)

            if meas:
                plt.scatter(x, y, label="meas")

            plt.scatter(x, y_pred, label="pred")

            plt.legend()
            plt.tight_layout()

            plt.savefig(f"{prefix}power_curve_{turbine}.png", dpi=300)
            plt.close()


def get_wind_direction(u: float, v: float):
    norm_vec = np.array([0, 1])
    uv_vec = np.array([u, v])

    rot_angle = -270 / 180 * np.pi
    rot_mat = np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )

    uv_vec = rot_mat @ uv_vec

    angle = (
        np.arccos(
            np.dot(norm_vec, uv_vec)
            / (np.linalg.norm(norm_vec) * np.linalg.norm(uv_vec))
        )
        / np.pi
        * 180
    )

    return angle if v >= 0 else angle + (180 - angle) * 2


def plot_power_diff(
    model,
    dataset,
    wtg_1: int,
    wtg_2: int,
    meas: bool = False,
    prefix: str | None = None,
):
    """wtg_1 is considered the upstream turbine and wtg_2 the downstream one."""
    with torch.no_grad():
        y_pred_1 = []
        y_pred_2 = []
        y_1 = []
        y_2 = []
        x = []
        for data in dataset:
            y_pred_1.append(
                model(data.x, data.edge_index, data.edge_attr)
                .detach()
                .numpy()[wtg_1]
            )
            y_pred_2.append(
                model(data.x, data.edge_index, data.edge_attr)
                .detach()
                .numpy()[wtg_2]
            )

            if meas:
                y_1.append(data.y[wtg_1].numpy())
                y_2.append(data.y[wtg_2].numpy())

            u = data.x.numpy()[wtg_1, 0]
            v = data.x.numpy()[wtg_1, 1]

            wind_direction = get_wind_direction(u, v)

            x.append(wind_direction)

        x_arr = np.asarray(x)
        sorted_x_inds = np.argsort(x_arr)
        y_pred_diff = np.array(y_pred_2) - np.array(y_pred_1)
        y_pred_roll = (
            pd.Series(y_pred_diff.flatten()[sorted_x_inds])
            .rolling(20, min_periods=1)
            .mean()
            .to_numpy()
        )

        if meas:
            y_meas_diff = np.array(y_2) - np.array(y_1)
            y_meas_roll = (
                pd.Series(y_meas_diff.flatten()[sorted_x_inds])
                .rolling(20, min_periods=1)
                .mean()
                .to_numpy()
            )

            plt.scatter(x, y_meas_diff, label="meas")
            plt.plot(
                x[sorted_x_inds],
                y_meas_roll,
                label="meas_roll",
                color="orange",
            )

        plt.scatter(x, y_pred_diff, label="pred")
        plt.plot(x[sorted_x_inds], y_pred_roll, label="pred_roll", color="red")

        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{prefix}power_diff_{wtg_1}_{wtg_2}.png", dpi=300)
        plt.close()


def attention_weights(attention_tuple: tuple, dataset):
    g = utils.to_networkx(dataset[0], to_undirected=False)

    pos = dict(
        zip(
            # [entry["id"] for entry in config["windfarm"]["turbines"]],
            [entry for entry in range(6)],
            dataset[0].pos.tolist(),
        )
    )

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    edge_inds = zip(
        attention_tuple[0][0].tolist(), attention_tuple[0][1].tolist()
    )

    A = attention_tuple[1].detach().numpy()
    Aheat = np.ones((6, 6)) * -1
    for edge, (i, j) in enumerate(edge_inds):
        g.add_edge(i, j, weight=f"{A[edge, 0]:.5f}")
        Aheat[i, j] = A[edge, 0]

    labels = nx.get_edge_attributes(g, "weight")
    nx.draw(g, pos, with_labels=True, ax=ax, alpha=0.4, node_size=200)
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=labels,
        ax=ax,
        label_pos=0.3,
        verticalalignment="center_baseline",
        # horizontalalignment="right",
        font_size=20,
        rotate=False,
    )
    plt.tight_layout()

    plot_path = Path("experiment_plots/graph.png")
    plt.savefig(plot_path, dpi=300)

    plt.close()

    sns.heatmap(Aheat, annot=True, fmt=".5f")
    plt.tight_layout()

    plot_path = Path("experiment_plots/heatmap.png")
    plt.savefig(plot_path, dpi=300)

    plt.close()
