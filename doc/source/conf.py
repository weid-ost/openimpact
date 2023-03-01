# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import re
from pathlib import Path

import yaml

src_path = Path("../..").resolve()
sys.path.insert(0, str(src_path))
# -- Project information -----------------------------------------------------
metadata_name = "version.yaml"
metadata_filepath = src_path / "openimpact-deploy" / metadata_name
if not metadata_filepath.exists():
    raise RuntimeError(f"Configuration file {metadata_filepath} found!")

with metadata_filepath.open(mode="r") as f:
    meta_ = yaml.load(f, Loader=yaml.FullLoader)

project = meta_["pkg-name"]
copyright = "Copyright (C) " + meta_["copyright"]
author = meta_["author"]
# The full version, including alpha/beta/rc tags
release = meta_["version"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
rendermath = (
    "sphinx.ext.mathjax" if "MATHIMG" not in os.environ else "sphinx.ext.imgmath"
)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    rendermath,
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.autoyaml",
    "sphinx_autodoc_typehints",
    "sphinx_autodoc_defaultargs",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for PDF output -------------------------------------------------

# latex_engine = 'pdflatex'
latex_engine = "xelatex"
# latex_use_xindy = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinxdoc"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

# -- Options for Autodoc extension ----------------------------------------------
# This value selects if automatically documented members are sorted alphabetical
# (value 'alphabetical'), by member type (value 'groupwise') or by source order
# (value 'bysource'). The default is alphabetical.
autodoc_member_order = "groupwise"

# -- Options for Napoleon extension ----------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
# -- Options for autosectionlabel extension --------------------------------------
# Make sure the target is unique
autosectionlabel_prefix_document = True
# -- Options for typehints extension ---------------------------------------------
typehints_document_rtype = True
typehints_use_rtype = True
# -- Options for autoyaml extension ----------------------------------------------
autoyaml_level = 9999
# -- Options for Intersphinx extension ----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}
# -- Options for Gallery extension ----------------------------------------------
sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": [
        "../../examples",
    ],
    "only_warn_on_example_error": True,
    # files executed
    "filename_pattern": f"{re.escape(os.sep)}example_",
    # excluded files
    #'ignore_pattern': f'{re.escape(os.sep)}(?!example_)'
}
# -- Options for Math rendering ----------------------------------------------
imgmath_image_format = "svg"
imgmath_use_preview = True
imgmath_font_size = 14
# -- Options for Default arguments extension ----------------------------------
rst_prolog = (
    '''
.. |default| raw:: html

    <div class="default-value-section">'''
    + ' <span class="default-value-label">Default:</span>'
)
# -- Post process ------------------------------------------------------------
## Do not show default documentation of named tuples
import collections


def remove_namedtuple_attrib_docstring(app, what, name, obj, skip, options):
    if type(obj) is collections._tuplegetter:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", remove_namedtuple_attrib_docstring)
