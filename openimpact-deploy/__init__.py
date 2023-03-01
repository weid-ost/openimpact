"""Module to setup openimpact-deploy"""

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

# Author: Juan Pablo Carbajal <juanpablo.carbajal@ost.ch>

import yaml
import importlib
import warnings
from importlib.metadata import metadata, files
from pathlib import Path
from textwrap import dedent, indent

try:
    # Import this package
    # this adds it to the path so metadata works
    module_ = importlib.import_module(__name__)

    # Cast Message to Dictionary
    meta_ = dict(metadata(__name__).items())

    # Use lowercase keys
    meta_ = {k.lower(): v for k, v in meta_.items()}
    # extract copyright from first line of license string
    lic_lines_ = meta_["license"].split("\n")
    meta_["copyright"] = lic_lines_[0]
    meta_["license"] = "\n".join(lic_lines_[1:])
    del module_, lic_lines_

except importlib.metadata.PackageNotFoundError:
    warnings.warn(
        f"The package '{__name__}' was not found by importlib! "
        "Reading metadata from file"
    )

    __metadata_name__ = "version.yaml"
    __metadata_filepath__ = Path(__file__).parent.resolve() / __metadata_name__
    if not __metadata_filepath__.exists():
        raise ImportError("No package metadata file found!")

    with __metadata_filepath__.open(mode="r") as f:
        meta_ = yaml.load(f, Loader=yaml.FullLoader)

    meta_["copyright"] = "Copyright (C) " + meta_["copyright"]

# Add are fields in metadata to module's privates
__version__ = meta_["version"]
__author__ = meta_["author"]
__license__ = meta_["license"]
__copyright__ = meta_["copyright"]

del meta_

# Load configuration file from current folder or from package files if none in cwd
# if package is not installed load from the folder of this file
__config_name__ = f"{__name__}_config.yaml"
__config_filepath__ = Path.cwd() / __config_name__
if not __config_filepath__.exists():
    try:
        pkg_files = files(__name__)
        cfg_path_ = list(filter(lambda x: x.name == __config_name__, pkg_files))
        if cfg_path_:
            __config_filepath__ = Path(cfg_path_[0].locate())
            del pkg_files, cfg_path_
        else:
            del pkg_files, cfg_path_
            raise ImportError("No configuration file found!")

    except (importlib.metadata.PackageNotFoundError, ImportError):
        # If package is not installed
        __config_filepath__ = Path(__file__).parent / __config_name__

with __config_filepath__.open(mode="r") as f:
    #: Dictionary containing the data in the configuration file.
    #: The configuration file (fields of this dictionary) is documented in
    #: :ref:`package-config`.
    #:
    #: :meta hide-value:
    rcConfig = yaml.load(f, Loader=yaml.FullLoader)


# Make simple function to check package installation
def describe():
    """Print package information."""
    from rich.console import Console
    from rich.syntax import Syntax

    console = Console()
    print = console.print

    print("\n")
    print(f"[b]Package[/b]: {__name__} [bold cyan]{__version__}[/bold cyan]")
    print(f"[b]Author[/b]: {__author__}")
    console.rule("[bold red] License")
    print(__copyright__ + "\n", style="bold")
    print(__license__)

    console.rule("[bold red] Configuration")
    print(f"[b]File[/b]: {__config_filepath__}")
    print("[b]Contents:[/b]")
    yml = [
        l
        for l in __config_filepath__.read_text().split("\n")
        if not l.startswith("#") and l
    ]
    syntax = Syntax(indent("\n".join(yml), " " * 4), "yaml")
    print(syntax)
    print("\n")
