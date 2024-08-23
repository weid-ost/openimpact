"""
    Wranglers for transforming various data sources to the OpenIMPACT data format.

    Configuration file
    --------------------------
    Configuration file to map data source files to OpenIMPACT data format.

"""

# Copyright (C) 2023 OST Ostschweizer Fachhochschule
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Author: Florian Hammer <florian.hammer@ost.ch>

from pathlib import Path
import pandas as pd
from openimpact.utils import read_config


def to_openimpact(df: pd.DataFrame, *, column_mapping: dict) -> pd.DataFrame:
    """Transform dataframe containing raw data to an
    OpenIMPACT dataframe.

        To be used with the OpenIMPACT data pipeline.

        Parameters
        ----------
        df:
            DataFrame

        Returns
        -------
        OpenIMPACT dataframe
            The OpenIMPACT dataframe is defined in the OpenIMPACT data manifest and
            is used as a data interface for the OpenIMPACT data pipeline

        Example
        --------

    """

    df_renamed = df.rename(columns=column_mapping)[
        column_mapping.values()
    ].copy()

    return df_renamed


rcConfig = dict()
"""Default wrangler configuration.

  :meta hide-value:
"""


def set_default_cfg(cfg_file: str | Path):
    """Set a default configuration file."""
    global rcConfig
    rcConfig = read_config(cfg_file)


def _get_cfg_field(field: str, cfg_file: str | Path | None = None) -> dict:
    if cfg_file is not None:
        cfg = read_config(cfg_file)
        cfg = cfg[field].copy()
    else:
        # read form default or cached config
        cfg = rcConfig[field].copy()
    return cfg


def get_column_mapping(config_path: str | Path | None = None) -> dict:
    columns = _get_cfg_field("columns", config_path)
    return {col["name-from-source"]: col["name"] for col in columns}


def get_csv_fmt(config_path: str | Path | None = None) -> dict:
    return _get_cfg_field("csv", config_path)


def get_index_fmt(cfg_file: str | Path | None = None) -> dict:
    """Get Index format from configuration."""
    cfg = _get_cfg_field("index", cfg_file)
    return {
        "name_mapping": (cfg["name-from-source"], cfg["name"]),
        "dt_format": cfg["unit"],
        "tz_mapping": (cfg["time-zone-from-source"], cfg["time-zone"]),
    }


def get_dataset(config_path: str | Path | None = None) -> dict:
    return _get_cfg_field("dataset", config_path)


def read_raw(
    fpath: Path | str,
    *,
    csv_fmt: dict | None = None,
    index_fmt: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read raw data file.

    Parameters
    ----------
    fpath:
        Path to data file.
    csv_fmt:
        CSV format dictionary options to be passed to :func:`pandas.read_csv`.
        If not given, the output of :func:`~.get_csv_fmt` is used.
    index_fmt:
        Information about the index of the returned data.
        If not given, the output of :func:`~.get_index_fmt` is used.
    drop_unnamed:
        Whether to skip unnamed columns.

    Returns
    -------
    data:
        Formatted raw data.

    Examples
    --------
    """

    if csv_fmt is None:
        csv_fmt = get_csv_fmt()

    df = pd.read_csv(fpath, **csv_fmt, **kwargs)

    # Set index
    if index_fmt is None:
        index_fmt = get_index_fmt()

    df.rename(columns=dict([index_fmt["name_mapping"]]), inplace=True)
    df.set_index(index_fmt["name_mapping"][1], inplace=True)
    # parse dates
    df.index = pd.to_datetime(df.index, unit=index_fmt["dt_format"])
    # change time zone to UTC
    from_, to = index_fmt["tz_mapping"]
    df.index = df.index.tz_localize(from_).tz_convert(to)

    return df
