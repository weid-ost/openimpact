from pathlib import Path
from typing import Callable, Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

from openimpact.data.download import zenodo_download
from openimpact.data.graph_gen import (
    create_graph,
    update_states,
)


class KelmarshDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return ["Kelmarsh_WT_static.csv", "Kelmarsh_SCADA_2016_3082.zip"]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        """Downloads the Kelmarsh data from Zenodo
        Main URL: https://zenodo.org/record/7212475
        API URL: https://zenodo.org/api/records/7212475
        """
        zenodo_download(
            7212475,
            self.raw_dir,
            force_download=False,
            include=self.raw_file_names,
        )

    def process(self):
        df_static = pd.read_csv(
            f"{self.raw_dir}/Kelmarsh_WT_static.csv", compression="gzip"
        )

        wt_mapping = {
            "KWF1": 228,
            "KWF2": 229,
            "KWF3": 230,
            "KWF4": 231,
            "KWF5": 232,
            "KWF6": 233,
        }

        lat_lon_dict = {
            wt_mapping[wt]: (lat, lon)
            for wt, lat, lon in df_static[
                ["Alternative Title", "Latitude", "Longitude"]
            ].to_records(index=False)
        }

        DG = create_graph(lat_lon_dict, max_dist=1.0)

        scada_files = [
            f"{self.raw_dir}/{name}"
            for name in self.raw_file_names
            if "SCADA" in name
        ]

        df_ts = _read_scada(scada_files)

        state_columns = [
            "Wind speed (m/s)",
            "Wind direction (°)",
            "Nacelle position (°)",
        ]
        target_col = ["Power (kW)"]
        df_states = df_ts[["wt_id"] + state_columns + target_col].copy()

        times = np.random.choice(df_states.index.unique(), 300)

        graph_list = []

        for time in times:
            for state_name in state_columns:
                df_state = df_states.loc[time][["wt_id", state_name]]
                states = df_state.to_records(index=False)

                DG = update_states(DG, state_name, states)

            df_target = df_states.loc[time][["wt_id", target_col[0]]]
            target = df_target.to_records(index=False)
            DG = update_states(DG, "y", target)

            graph_list.append(DG)

        edge_attrs = ["dist", "dir"]

        data_list = []
        for graph in graph_list:
            edge_attributes = (
                edge_attrs if graph.number_of_edges() != 0 else None
            )
            data = from_networkx(
                graph,
                group_node_attrs=state_columns,
                group_edge_attrs=edge_attributes,
            )

            # TODO: Got some casting errors in graphgym. This is a quick fix.
            data.x = data.x.float()
            data.y = data.y.float()

            # TODO: Graph gym looks for the following attributes. This is a work around to avoid graphgym raising an exception
            data.train_mask = torch.ones(graph.number_of_nodes(), dtype=bool)
            data.val_mask = data.train_mask
            data.test_mask = data.train_mask

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


def _scada_zip_to_dataframe(filename: str | Path):
    with ZipFile(filename) as myzip:
        data_files = [f for f in myzip.namelist() if "Turbine_Data" in f]

        concat = []
        for f in data_files:
            path = Path(f)
            wt_id = int(path.stem.split("_")[-1])

            with myzip.open(f, "r") as wt:
                df_tmp = (
                    pd.read_csv(wt, skiprows=9)
                    .rename(columns={"# Date and time": "Datetime"})
                    .set_index("Datetime")
                )
                df_tmp["wt_id"] = wt_id
                df_tmp = df_tmp

                concat.append(df_tmp)

    return pd.concat(concat)


def _read_scada(files: list[str | Path]):
    concat = []
    for filename in files:
        concat.append(_scada_zip_to_dataframe(filename))

    return pd.concat(concat)


if __name__ == "__main__":
    dataset = KelmarshDataset("kelmarsh_test")
    print(dataset[0])
