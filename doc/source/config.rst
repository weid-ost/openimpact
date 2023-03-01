.. _package-config:

Package configuration
======================

The package can be configured using a YAML file named ``openimpact-deploy_config.yaml``.
The package comes with a default configuration file that can be found in the
installation folder of the package.
The current file that is used by the package (e.g. the default one) can be seen
at the end of the output of the function :func:`openimpact-deploy.describe`.
This function can be run as a terminal script via the command::

    openimpact-deploy-describe

after the package is installed in your system.

To change the configuration of the package you can copy this file to the folder
where you will be running the programs and then modify the values in it.
Some scripts might allow to override the value of some variables via
the command line.

Package configuration file
--------------------------

.. autoyaml:: ../openimpact-deploy/openimpact-deploy_config.yaml

.. hint::
   The information in the configuration can be accessed programmatically too
   through the :mod:`openimpact-deploy.configuration` abstraction. It can also be
   directly accessed via :data:`openimpact-deploy.rcConfig` dictionary in the main
   package, although this is not recommended.
