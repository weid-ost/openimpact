""" setup script for openimpact-deploy package """

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

import yaml
import datetime
from setuptools import setup, find_packages

# Load long description from README file
with open("README.rst", encoding="utf-8") as f:
    long_description = f.read()

# Load versioning metadata from yaml file
version_file = f"./openimpact-deploy/version.yaml"
with open(version_file, encoding="utf-8") as f:
    meta = yaml.load(f, Loader=yaml.FullLoader)

# Add copyright to license string
copyright_license = f"""Copyright (C) {datetime.datetime.now().year} {meta["copyright"]}
{meta["license"]}"""
# Join author information
authors = ", ".join(meta["author"]) if isinstance(meta["author"], list) else meta["author"]
emails = ", ".join(meta["email"]) if isinstance(meta["email"], list) else meta["email"]

setup(
    name=meta["pkg-name"],
    version=meta["version"],
    author=meta["author"],
    author_email=meta["email"],
    license=copyright_license,
    url=meta["url"],
    description=meta["description"],
    python_version=meta["python-version"],
    packages=find_packages(exclude=["tests*"]),
    # These are your install dependencies: everything you need to run production
    # code. Not for development of compressor_experiments.
    install_requires=[
        "pyaml",
        "rich",
        ""
    ],
    long_description=long_description,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            f"{meta['pkg-name']}-describe={meta['pkg-name']}:describe",
            # add other console entry points here
        ],
    },
)
