Installation
=============

Firstly, Apache Hadoop, Apache Spark, PostgreSQL and ``dbt-core`` need to be installed.

Apache Hadoop
~~~~~~~~~~~~~

Coming soon...


Apache Spark
~~~~~~~~~~~~

Download and extract the latest version of Spark:

.. code-block:: console

    $ wget https://www.apache.org/dyn/closer.lua/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
    $ tar xfv spark-3.4.0-bin-hadoop3.tgz

Set ``SPARK_HOME`` in your ``.bashrc`` or equivalent.

.. code-block:: console

    $ echo 'export SPARK_HOME="/home/florian/spark-3.4.0-bin-hadoop3"' >> ~/.bashrc

PostgreSQL
~~~~~~~~~~

Install `PostgreSQL`_

.. code-block:: console

    # Create the file repository configuration:
    $ sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'

    # Import the repository signing key:
    $ wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -

    # Update the package lists:
    $ sudo apt-get update

    # Install the latest version of PostgreSQL.
    # If you want a specific version, use 'postgresql-12' or similar instead of 'postgresql':
    $ sudo apt-get -y install postgresql

.. _PostgreSQL: https://www.postgresql.org/download/

Download the `PostgreSQL JDBC`_ drivers

.. code-block:: console

    $ wget https://jdbc.postgresql.org/download/postgresql-42.6.0.jar

.. _PostgreSQL JDBC: https://jdbc.postgresql.org/download/

Move PostgreSQL JDBC to Spark ``jars`` directory

.. code-block:: console

    $ mv postgresql-42.6.0.jar $SPARK_HOME/jars/

Check if Java is installed. If not, then install.

.. code-block:: console

    $ java -version

    # If the command was not found, install java
    $ sudo apt install default-jre

Check that ``JAVA_HOME`` is set. Otherwise, set it in your ``.bashrc`` or equivalent.

.. code-block:: console

    $ echo $JAVA_HOME

    # If the output is empty, set JAVA_HOME
    $ echo 'export JAVA_HOME="/path/to/java"' >> ~/.bashrc 


``dbt-core``
~~~~~~~~~~~~

Create a project folder for your ``dbt`` pipeline project

.. code-block:: console

    $ mkdir /path/to/dbt-project

Create virtual Python environment

.. code-block:: console

    $ cd /path/to/dbt-project
    $ python -m venv .venv
    $ source .venv/bin/activate

    # Check that the correct python binary is used
    (.venv) $ which python


Install ``dbt-core`` and ``pyspark``

.. code-block:: console

    (.venv) $ pip install dbt-core "dbt-spark[PyHive]" pyspark


