"""Generally used functions. """
from typing import Union
from pathlib import Path
from io import StringIO
import yaml


def yaml_load(file_path: Union[str, Path], *, ignore_private: bool = True) -> dict:
    """Load YAMl file, ignoring private fields (prefix: "__").

    Parameters
    ----------
    file_path:
        Path to YAML file.
    ignore_private:
        Flag to disable loading of private fields in the YAML file.

    Returns
    -------
    data:
        A dict containing the loaded key-value pairs from the YAML file.

    """
    with open(file_path, encoding="utf8") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        data = yaml.load(file, Loader=yaml.FullLoader)
        if data is None:
            data = {}
    if data:  # do nothing if empty
        if ignore_private:
            # erase private fields (starts with __)
            for k in tuple(k_ for k_ in data.keys() if str(k_).startswith("__")):
                data.pop(k)
    return data


try:
    from pandas import DataFrame

    def dataframe_info(df: DataFrame, *, verbose: bool = True) -> str:
        """Get string with dataframe summary information."""
        buffer = StringIO()
        df.info(buf=buffer, verbose=verbose)
        return buffer.getvalue()

except ImportError:
    pass
