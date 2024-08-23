API Reference
=============
The main openimpact API is split into the following packages:

* ``openimpact``
* ``openimpact.model``
* ``openimpact.data``


``openimpact.model``
--------------------

Layers and GNNs
~~~~~~~~~~~~~~~

.. autosummary::

   ~openimpact.model.gnn.GATLayer
   ~openimpact.model.gnn.BaseGNN
   ~openimpact.model.gnn.FarmGNN
   ~openimpact.model.gnn.FarmGNN3
   ~openimpact.model.gnn.FarmGAT
   ~openimpact.model.gnn.get_checkpoint
   ~openimpact.model.gnn.load_model

Functions
~~~~~~~~~

.. autosummary::

   ~openimpact.model.features
   ~openimpact.model.hparam_tune
   ~openimpact.model.train


``openimpact.data``
--------------------

Utility functions
~~~~~~~~~~~~~~~~~

.. autosummary::

   ~openimpact.data.wranglers
   ~openimpact.data.preprocessing
   ~openimpact.data.download
   ~openimpact.data.data_store
   ~openimpact.data.datasets.graph_gen
   ~openimpact.data.datasets.distance
   ~openimpact.data.datasets.utils

Datasets
~~~~~~~~

.. autosummary::

   ~openimpact.data.datasets.kelmarsh.KelmarshDataset

Source code 
-----------

GNN
~~~

.. automodule:: openimpact.model.gnn
   :members:
   :undoc-members:

Datasets
~~~~~~~~

.. automodule:: openimpact.data.datasets.kelmarsh
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.data.datasets.graph_gen
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.data.datasets.distance
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.data.datasets.utils
   :members:
   :undoc-members:
   :ignore-module-all:

Data
~~~~

.. automodule:: openimpact.data.wranglers
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.data.data_store
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.data.download
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.data.preprocessing
   :members:
   :undoc-members:
   :ignore-module-all:

Model
~~~~~

.. automodule:: openimpact.model.hparam_tune
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.model.train
   :members:
   :undoc-members:
   :ignore-module-all:

.. automodule:: openimpact.model.features
   :members:
   :undoc-members:
   :ignore-module-all:

