==================================
Python Package openimpact-deploy
==================================

Placeholder for brief description



main:|pipelineStatusMain|_ release: |pipelineStatusRelease|_ |latestRelease|_

|testCoverage|_

.. |pipelineStatusMain| image:: https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy/badges/main/pipeline.svg
.. _pipelineStatusMain: https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy/-/pipelines

.. |pipelineStatusRelease| image:: https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy/badges/release/pipeline.svg
.. _pipelineStatusRelease: https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy/-/pipelines

.. |latestRelease| image:: https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy/-/badges/release.svg
.. _latestRelease: https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy/-/releases

.. |testCoverage| image:: https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy/badges/main/coverage.svg?job=coverage
.. _testCoverage: http://windenergyatost.pages.gitlab.ost.ch/openimpact/openimpact-deploy/coverage



.. contents:: :local:

Documentation
--------------------
The documentation is hosted at <http://windenergyatost.pages.gitlab.ost.ch/openimpact/openimpact-deploy/doc>_.

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

::

    pip install "git+https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy.git"

To install the release

::

   pip install "git+https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy.git@release"

If you want a specific release replace the word release with the version tag

::

   pip install "git+https://gitlab.ost.ch/windenergyatost/openimpact/openimpact-deploy.git@v0.2.0"


Alternatively, if you have downloaded the library into a folder ``LIBFOLDER`` you can run

::

    pip install LIBFOLDER

You can check the installation of the package by running::

    python -c 'import openimpact-deploy; openimpact-deploy.describe()'

which will print some basic information about the package.
Also check if access points are correctly configured. The command::

    openimpact-deploy-describe

should produce the same output as the previous command.

.. hint::
    The package was developed using Python latest stable distribution.


Development
--------------------

To develop the library it is best to create a virtual environment.
See `Virtual environments`_ below if you need instructions to set one.

.. warning::
   The following instructions will assume that you have created and activated the
   virtual environment, and installed the latest version of `pip` and `wheel`.

To develop the library you need to install the development requirements::

    pip install --upgrade -r requirements.txt

Folder structure
*********************

The repository has the following structure::

    .
    ├── LICENSE
    ├── MANIFEST.in
    ├── pyproject.toml
    ├── README.rst
    ├── requirements.txt
    ├── setup.py
    ├── openimpact-deploy
    │   ├── __init__.py
    │   ├── openimpact-deploy_config.yaml
    │   ├── version.yaml
    │   └── scripts
    │       └── ...
    ├── doc
    │   ├── source
    │   │   └── ...
    │   └── ...
    ├── examples
    │   ├── data
    │   │   └── ...
    │   └── ...
    └── tests
        ├── data
        │   └── ...
        └── ...

``openimpact-deploy``
    Contains the source code of the package.

    The file ``_init__.py`` defines maintainers functionality, and the
    ``describe`` function used by the ``openimpact-deploy-describe`` entry point (script).

    The file ``openimpact-deploy_config.yaml`` is the default configuration of the package.

    The file ``version.yaml`` contains versioning information.

    The folder ``scripts`` contains runnable scripts that give quick access to some
    of the library's functionalities.
    These scripts are also provided as command line entry points with the naming
    scheme ``openimpact-deploy-<script-name>``.
    Type ``openimpact-deploy-<script-name> --help`` to get more information about running a
    script.

``doc``, ``examples``
    Folder use to document the library.

    The folder ``doc`` is a `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_
    documentation folder, which is used to configure the generated documentation.
    The contents for the documentation are placed in the sub-folder ``source``.

    The folder ``examples`` contain runnable scripts that illustrate the use of
    the library.
    These examples are also included in the generated documentation.
    The examples are included in the documentation via the Sphinx extension
    `Sphinx-Gallery <https://sphinx-gallery.github.io/stable/index.html>`_

``tests``
    Contains API tests (not exhaustive) and verification tests.
    These are run using `unittest <https://docs.python.org/3/library/unittest.html>`_

Unit tests
************
The quickest way to run all tests is to execute::

        python -m unittest -f

in the root folder of the cloned repository after you have installed all dependencies.

Virtual environments
*********************
To avoid collisions with your system's library versions, use a python virtual environment for installation.
We recommend that you use standard python virtual environments as provided by `venv <https://docs.python.org/3/library/venv.html>`_.
Also, we recommend to use the system's terminal (e.g. Powershell in Windows).
If you are running with an specialized tools, e.g. Anaconda, and you are using a terminal provided by the tool you can still use the standard tools.

1. Create a virtual environment in the desired path, let's call it ``MYPATH``::

        python -m venv MYPATH/openimpact-deploy


2. Activate the environment

   In a Linux bash terminal::

        source MYPATH/openimpact-deploy/bin/activate


   In an Anaconda PowerShell prompt or on a Windows PowerShell::

        MYPATH/openimpact-deploy/Scripts/Activate.ps1

3. Update pip and wheel::

        pip install -U pip wheel

Code coverage
***************

Code coverage report can be found at http://windenergyatost.pages.gitlab.ost.ch/openimpact/openimpact-deploy/coverage.
