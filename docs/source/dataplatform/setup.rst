Setup
======

For Apache Hadoop, Apache Spark, PostgreSQL and ``dbt-core`` to work together, the following steps have to be executed.

Apache Hadoop
~~~~~~~~~~~~~~

.. code-block:: console

    $ export PDSH_RCMD_TYPE=ssh


In ``etc/hadoop/hadoop-env.sh`` you also need to set ``JAVA_HOME``.

.. code-block:: console

    export JAVA_HOME=/path/to/java

PostgreSQL
~~~~~~~~~~

Create a PostgreSQL database and a user, which has all the rights to create tables.

.. code-block:: console

    $ sudo -u postgres psql
    postgres=# CREATE DATABASE hive;
    postgres=# CREATE USER hive WITH PASSWORD 'hive';
    postgres=# \q
    $ sudo -u postgres psql -d hive
    hive=# GRANT CREATE ON SCHEMA public TO hive;
    hive=# \q

The choice of having a user and database named ``hive`` is arbitrary. Any database name and user can be chosen.
However, in the following sections the user and database name ``hive`` is used within configuration files and needs to be replaced in case some other names are used.

Apache Spark
~~~~~~~~~~~~

Create a ``warehouse`` directory for the Spark / Hive Data Warehouse

.. code-block:: console

    $ mkdir /path/to/project/warehouse

Create ``hive-site.xml`` with following content

.. code-block:: xml

    <configuration>
        <property>
            <name>javax.jdo.option.ConnectionURL</name>
            <value>jdbc:postgresql://localhost:5432/hive</value>
        </property>
        <property>
            <name>hive.metastore.warehouse.dir</name>
            <value>/path/to/project/warehouse</value>
            <description>location of Hive warehouse directory</description>
        </property>
        <property>
            <name>datanucleus.schema.autoCreateTables</name>
            <value>true</value>
        </property>
        <property>
            <name>javax.jdo.option.ConnectionDriverName</name>
            <value>org.postgresql.Driver</value>
        </property>
        <property>
            <name>javax.jdo.option.ConnectionUserName</name>
            <value>hive</value>
        </property>
        <property>
            <name>javax.jdo.option.ConnectionPassword</name>
            <value>hive</value>
        </property>
        <property>
            <name>hive.metastore.schema.verification</name>
            <value>false</value>
        </property>
    </configuration>

Save the file in the Spark ``conf`` directory

.. code-block:: console

    $ mv hive-site.xml $SPARK_HOME/conf/


Launch Spark and start the Thrift JDBC Server
----------------------------------------------

In order to start Spark Thrift Server, change to the ``warehouse`` directory and run the following sheel scripts

.. code-block:: console

    $ cd /path/to/project/warehouse
    $ $SPARK_HOME/sbin/start-master.sh --host 0.0.0.0
    $ $SPARK_HOME/sbin/start-slave.sh spark://localhost:7077
    $ $SPARK_HOME/sbin/start-thriftserver.sh --total-executor-cores 2 --master spark://localhost:7077

This will first start a standalone master server on ``localhost`` and then a worker instance. 
Information about the server can be found in the Web UI on http://localhost:8080.
After that the Spark Thrift Server is started, using the resources from the worker instance.
The Spark Thrift Server is 

    Spark SQL's port of Apache Hive's HiveServer2 that allows JDBC/ODBC clients to execute SQL queries over JDBC and ODBC protocols on Apache Spark.

Extracting and loading data with ``PySpark`` into the Spark warehouse as well as running your ``dbt`` pipeline will be done over the JDBC protocol on the Apache Spark server.