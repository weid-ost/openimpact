"""Abstraction layer for configuration file of openimpact-deploy.

Write functions that retrieve data from the rcConfig dictionary.
This prevents that your program is aware of the structure of the configuration
file. 
Only this module needs to be aware of it, thus make a single point of edition
if you change the configuration file.

For example, if you have an `the-url` field defined in your config file, write a function
like this one::

    def url() -> str:
        return rcConfig["the-url"]

In the configuration file document the field like this::

    ###
    # Explanation of what url is.
    #
    # Detailed explanation of what url is.
    #
    # Access function: :func:`openimpact-deploy.configuration.url`

See `sphinx-autoyaml <https://pypi.org/project/sphinxcontrib-autoyaml>`_ for details about documenting a YAML file.
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

import logging

from . import __config_filepath__
from . import utils

logger = logging.getLogger(__name__)

#: Dictionary containing the data in the configuration file.
#: The configuration file (fields of this dictionary) is documented in
#: :ref:`package-config`.
#:
#: :meta hide-value:
rcConfig = utils.yaml_load(__config_filepath__)
