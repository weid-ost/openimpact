from openimpact.data.wranglers import (
    set_default_cfg,
    get_column_mapping,
    get_csv_fmt,
    get_index_fmt,
    get_dataset,
)
from openimpact.utils import read_config
import pytest


class TestConfig:
    set_default_cfg("tests/default.toml")

    def test_read_cfg(self):
        with pytest.raises(FileNotFoundError):
            config = read_config("not_exist.toml")
        config = read_config("tests/default.toml")
        assert config

    def test_column_mapping(self):
        column_mapping = {
            "Wind speed": "wind_speed",
            "Power": "power",
            "Wind turbine ID": "wt_id",
        }
        assert column_mapping == get_column_mapping()

    def test_csv_fmt(self):
        csv_fmt = {
            "encoding": "utf8",
            "sep": ",",
            "header": 0,
        }
        assert csv_fmt == get_csv_fmt()

    def test_index_fmt(self):
        index_fmt = {
            "name_mapping": ("# Date and time", "datetime"),
            "dt_format": "ns",
            "tz_mapping": ("UTC", "UTC"),
        }

        assert index_fmt == get_index_fmt()

    def test_dataset(self):
        dataset = {
            "name": "default_dataset",
            "data": "default_data.csv",
            "static": "default_static.csv",
        }

        assert dataset == get_dataset()
