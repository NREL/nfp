============
Installation
============

NFP can be installed with pip into a suitable python environment with

.. code-block:: bash

   pip install nfp


The tensorflow and rdkit dependencies can be tricky to install. A recommended conda environment is

.. code-block:: yaml

    channels:
      - conda-forge
      - defaults

    dependencies:
      - rdkit
      - pytest
      - tqdm
      - pip
      - pip:
          - tensorflow>=2.0
