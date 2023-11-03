==================================
Python Package openimpact
==================================

Documentation
--------------------

To generate the documentation in a local copy of this repository,
do the following::

    pip install --upgrade -r requirements.txt

then::

    cd doc
    make html

This is how the online documentation is built.
The documentation will be generated in ``build/html/``

To embed mathematical formulas as images in the resulting html define the
environment variable ``MATHIMG``, e.g.::

    MATHIMG=1 make html

will generate math as images. This requires the program dvisvgm to be installed
, see https://www.sphinx-doc.org/en/master/usage/extensions/math.html for further
information.

Installation
--------------------
.. hint::
    To avoid collisions with your system's library versions,
    use a python virtual environment for installation. See
    `Virtual environments`_ below.

You can install the library from its repository

