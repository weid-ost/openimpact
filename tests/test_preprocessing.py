import pandas as pd
import pytest


from openimpact.data.preprocessing import (
    dropna,
    filter_bin,
    filter_power,
    match_group_count,
    rename_labels,
)
from openimpact.data.wranglers import set_default_cfg


class TestPreprocessing:
    set_default_cfg("tests/default.toml")

    def test_filter(self, openimpact_dataframe, clean_dataframe):
        df = dropna(openimpact_dataframe, axis=0)
        assert df.isna().sum().sum() == 0

        df = filter_power(df, "power", 1000, thresh=0.02)

        assert df.query("power < 20.0").shape[0] == 0

        pd.testing.assert_frame_equal(df, clean_dataframe)

    # def test_filter_bin(self, clean_dataframe):
    #     df = filter_bin(clean_dataframe, "wind_speed", "power", 2.0, 3.0, 15.0)
    #
    #     print(df.shape)
    #     assert df.shape == (92, 3)

    def test_match_group_count(self, clean_dataframe):
        df = match_group_count(
            clean_dataframe, by="datetime", count_col="wt_id", match=2
        )
        pd.testing.assert_frame_equal(df, clean_dataframe)

    def test_rename_labels(self, clean_dataframe):
        wt_mapping = {
            1: "KWF1",
            2: "KWF2",
        }

        s = rename_labels(clean_dataframe["wt_id"], wt_mapping)

        pd.testing.assert_series_equal(
            s, clean_dataframe["wt_id"].apply(lambda x: wt_mapping[x])
        )

    def test_freestream_conditions(self, clean_dataframe):
        pass
        # df = add_freestream_conditions(df)
