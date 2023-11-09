import pytest

from openimpact.data.wranglers import (
    to_openimpact,
    get_column_mapping,
    set_default_cfg,
)


class TestWrangler:
    set_default_cfg("tests/default.toml")

    def test_to_openimpact(self, raw_dataframe, openimpact_dataframe):
        column_mapping = get_column_mapping()
        with pytest.raises(TypeError):
            df = to_openimpact(raw_dataframe, column_mapping)

        df = to_openimpact(raw_dataframe, column_mapping=column_mapping)

        assert df.shape[1] == openimpact_dataframe.shape[1]
        assert set(df.columns).issubset(openimpact_dataframe.columns)
        assert df.index.name == openimpact_dataframe.index.name
