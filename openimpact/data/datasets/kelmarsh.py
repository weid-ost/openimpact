from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx

from .graph_gen import (
    create_graph,
    update_states,
)


class KelmarshDataset(InMemoryDataset):
    """
    Kelmarsh Dataset
    """

    def __init__(
        self,
        root,
        data_path: str | Path | None = None,
        windfarm_static_path: str | Path | None = None,
        features: list[str] | None = None,
        target: str | None = None,
        wt_col: str | None = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.data_path = data_path
        self.windfarm_static_path = windfarm_static_path

        self.features = features
        self.target = target
        self.wt_col = wt_col

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str | Path]:
        return [self.windfarm_static_path, self.data_path]

    @property
    def processed_file_names(self):
        return "data.pt"

    def process(self):
        df_static = pd.read_csv(self.windfarm_static_path)

        lat_lon_dict = {
            wt: (lat, lon)
            for wt, lat, lon in df_static[
                ["Alternative Title", "Latitude", "Longitude"]
            ].to_records(index=False)
        }

        DG = create_graph(lat_lon_dict, max_dist=1.0)

        df_ts = pd.read_parquet(self.data_path)
        # df_ts.set_index("datetime", inplace=True)

        df_states = df_ts[[self.wt_col] + self.features + [self.target]]

        times = df_states.index.unique()

        graph_list = []

        for time in times:
            for state_name in self.features:
                df_state = df_states.loc[time][[self.wt_col, state_name]]
                states = df_state.to_records(index=False)

                DG = update_states(DG, state_name, states)

            df_target = df_states.loc[time][[self.wt_col, self.target]]
            target = df_target.to_records(index=False)
            DG = update_states(DG, "y", target)

            graph_list.append(DG)

        # TODO: Get rid of coupling somehow. create_graph() creates x_dist and y_dist
        # and returns an nx.Graph containing those in the dict of attribute _adj.
        # Maybe there is a better way to set them here or figure out what their
        # names are based on the nx.Graph instance.
        edge_attrs = ["x_dist", "y_dist"]

        data_list = []
        for graph in graph_list:
            edge_attributes = (
                edge_attrs if graph.number_of_edges() != 0 else None
            )
            data = from_networkx(
                graph,
                group_node_attrs=self.features,
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
