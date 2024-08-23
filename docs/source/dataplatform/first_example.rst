Create your first ``dbt`` pipeline
==================================


Go to your ``dbt`` project folder and run ``dbt init``

.. code-block:: console

    $ cd /path/to/dbt-project

    # Active the python environment
    $ source .venv/bin/activate
    (.venv) $ dbt init

Enter your project name, e.g. ``first_pipeline``, 
select ``spark`` database, 
use ``localhost`` as the host, 
select the ``thrift`` authentication method with port ``10000``, 
enter a name for the default schema, e.g. ``public`` and 
finally set the number of threads, e.g. ``1``.


This will create a file ``profiles.yml`` in ``$HOME/.dbt`` and should look like this

.. code-block:: yaml

    first_pipeline:
        outputs:
            dev:
            host: localhost
            method: thrift
            port: 10000
            schema: public
            threads: 1
            type: spark
        target: dev

The command will also create a new folder ``first_pipeline`` with the following contents

..  code-block:: bash

    first_pipeline
    ├── snapshots
    ├── analyses
    ├── seeds
    ├── models
    ├── macros
    ├── README.md
    ├── tests
    └── dbt_project.yml



1. Run "dbt seed" to load static csv files

    .. code-block:: console

        (.venv) $ dbt seed

Good use-cases for seeds:

- A list of mappings of country codes to country names
- A list of test emails to exclude from analysis
- A list of employee account IDs
    
Poor use-cases of dbt seeds:

- Loading raw data that has been exported to CSVs
- Any kind of production data containing sensitive information. For example personal identifiable information (PII) and passwords.

5. Go to dbt project and run "dbt run"

    .. code-block:: console

        (.venv) $ dbt run



.. code-block:: python
    :emphasize-lines: 4,5

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("my_app")
            .config("spark.sql.warehouse.dir", "/path/to/warehouse",)    
            .config("url", "jdbc:hive2://localhost:10000")
            .enableHiveSupport()
            .getOrCreate()


.. code-block:: yaml
    :name: Project properties and config
    :caption: dbt_project.yml

    # Name your project! Project names should contain only lowercase characters
    # and underscores. A good package name should reflect your organization's
    # name or the intended use of these models
    name: "first_pipeline"
    version: "1.0.0"
    config-version: 2

    # This setting configures which "profile" dbt uses for this project.
    profile: "first_pipeline"

    # These configurations specify where dbt should look for different types of files.
    # The `model-paths` config, for example, states that models in this project can be
    # found in the "models/" directory. You probably won't need to change these!
    model-paths: ["models"]
    analysis-paths: ["analyses"]
    test-paths: ["tests"]
    seed-paths: ["seeds"]
    macro-paths: ["macros"]
    snapshot-paths: ["snapshots"]

    target-path: "target" # directory which will store compiled SQL files
    clean-targets: # directories to be removed by `dbt clean`
    - "target"
    - "dbt_packages"

    # Configuring models
    # Full documentation: https://docs.getdbt.com/docs/configuring-models

    # In this example config, we tell dbt to build all models in the example/ directory
    # as tables. These settings can be overridden in the individual model files
    # using the `{{ config(...) }}` macro.
    models:
    first_pipeline:
        # Config indicated by + and applies to all files under models/example/
        staging:
        +materialized: view
        intermediate:
        +materialized: ephemeral
        marts:
        +materialized: table
        +schema: windfarm


.. code-block:: yaml
    :name: dbt packages to be installed
    :caption: packages.yml

    packages:
      - package: dbt-labs/dbt_utils
        version: [">=0.8.0", "<0.9.0"]
      - package: dbt-labs/spark_utils
        version: [">=0.3.0", "<0.4.0"]
      - package: calogica/dbt_date
        version: [">=0.5.0", "<0.6.0"]

.. code-block:: console

    $ dbt deps


Apache Hadoop
-------------

Apache Hadoop is used as the Data Lake solution, as it Open Source and allows to run on commodity hardware.

    HDFS can be accessed from applications in many different ways. 
    Natively, HDFS provides a FileSystem Java API for applications to use. 
    A C language wrapper for this Java API and REST API is also available. 
    In addition, an HTTP browser and can also be used to browse the files of an HDFS instance. 
    By using NFS gateway, HDFS can be mounted as part of the client's local file system.


Furthermore,

    Important: all production Hadoop clusters use Kerberos to authenticate callers and secure access to HDFS data as well as restriction access to computation services (YARN etc.).


WESC 2023
=========

Ingestion, processing and storage
----------------------------------

.. code-block:: python

    from pyspark.sql import SparkSession

    spark = (
    SparkSession.builder.appName("PySpark - Data Lake and Data Warehouse")
    .config("spark.sql.warehouse.dir", "/path/to/warehouse",)
    .config("url", "jdbc:hive2://localhost:10000")
    .master("spark://0.0.0.0:7077")
    .enableHiveSupport()
    .getOrCreate()
    )

    path = "path/to/data.csv"
    data = spark.read.csv(path)

    data.write.format("parquet").save("hdfs://localhost:9000/path/to/data-parquet")

    # Create structured dataset

    spark.sql(f"USE windfarm")
    data_struct.write.mode("overwrite").saveAsTable("structured_data")


Processing and serving
----------------------

.. code-block:: python

    from pyspark.sql import SparkSession

    spark = (
    SparkSession.builder.appName("PySpark - Processing")
    .config("spark.sql.warehouse.dir", "/path/to/warehouse",)
    .config("url", "jdbc:hive2://localhost:10000")
    .master("spark://0.0.0.0:7077")
    .enableHiveSupport()
    .getOrCreate()
    )


    data = spark.read.parquet("hdfs://localhost:9000/path/to/data-parquet")

    # Filter, drop, rename, etc.
    data.filter("wind_speed > 0.5")

    # Create a LinearRegression model
    lr = LinearRegression(featuresCol='features', labelCol='label')

    # Fit the model to the training data
    lr_model = lr.fit(data)


    # Load processed data into data lake or warehouse
    data.write.format("parquet").save("hdfs://localhost:9000/path/to/processed-data")

    # Save file to local folder
    data.toPandas().to_csv("local_data.csv")


API
---

.. code-block:: python

    import pyarrow.fs as fs

    class WindFarmDataset(InMemoryDataset):

        def __init__(
            self,
            root,
            url,
            filename,
            config,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
        ):
            self.url = url
            self.filename
            self.config = config
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])

        def download(self):
        """Load data from data lake, data warehouse or local path"""

            hdfs = fs.HadoopFileSystem(self.url)
            self.data = hdfs.open_input_file(self.filename)


    url = "hdfs://localhost:9000"
    
    filename = "/path/to/processed-data"
    # or
    filename = "/path/to/local_file"

    dataset = WindFarmDataset(root=root, url=url, filename=filename, config=config)

    model = FarmGAT(dataset.num_node_features, h_dim=12).double()

    loader = DataLoader(train_dataset, batch_size=1)

    model.train()

    losses = []
    for epoch in range(800):
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out, data.y.reshape(data.num_nodes, -1))

            losses.append(loss.detach().numpy())

            loss.backward()
            optimizer.step()


