:orphan:

###############
Getting started
###############

Installation
============

Qiskit Machine Learning depends on the main Qiskit package which has its own
`Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__ detailing the
installation options for Qiskit and its supported environments/platforms. You should refer to
that first. Then the information here can be followed which focuses on the additional installation
specific to Qiskit Machine Learning.

Qiskit Machine Learning has some functions that have been made optional where the dependent code and/or
support program(s) are not (or cannot be) installed by default. Those are PyTorch and Sparse.
See :ref:`optional_installs` for more information.

.. tabbed:: Start locally

    The simplest way to get started is to follow the getting started 'Start locally' for Qiskit
    here `Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__

    In your virtual environment where you installed Qiskit simply add ``machine-learning`` to the
    extra list in a similar manner to how the extra ``visualization`` support is installed for
    Qiskit, i.e:

    .. code:: sh

        pip install qiskit[machine-learning]

    It is worth pointing out that if you're a zsh user (which is the default shell on newer
    versions of macOS), you'll need to put ``qiskit[machine-learning]`` in quotes:

    .. code:: sh

        pip install 'qiskit[machine-learning]'


.. tabbed:: Install from source

   Installing Qiskit Machine Learning from source allows you to access the most recently
   updated version under development instead of using the version in the Python Package
   Index (PyPI) repository. This will give you the ability to inspect and extend
   the latest version of the Qiskit Machine Learning code more efficiently.

   Since Qiskit Machine Learning depends on Qiskit, and its latest changes may require new or changed
   features of Qiskit, you should first follow Qiskit's `"Install from source"` instructions
   here `Qiskit Getting Started <https://qiskit.org/documentation/getting_started.html>`__

   .. raw:: html

      <h2>Installing Qiskit Machine Learning from Source</h2>

   Using the same development environment that you installed Qiskit in you are ready to install
   Qiskit Machine Learning.

   1. Clone the Qiskit Machine Learning repository.

      .. code:: sh

         git clone https://github.com/Qiskit/qiskit-machine-learning.git

   2. Cloning the repository creates a local folder called ``qiskit-machine-learning``.

      .. code:: sh

         cd qiskit-machine-learning

   3. If you want to run tests or linting checks, install the developer requirements.

      .. code:: sh

         pip install -r requirements-dev.txt

   4. Install ``qiskit-machine-learning``.

      .. code:: sh

         pip install .

   If you want to install it in editable mode, meaning that code changes to the
   project don't require a reinstall to be applied, you can do this with:

   .. code:: sh

      pip install -e .


.. _optional_installs:

Optional installs
=================

* **PyTorch**, may be installed either using command `pip install 'qiskit-machine-learning[torch]'` to install the
  package or refer to PyTorch [getting started](https://pytorch.org/get-started/locally/). When PyTorch
  is installed, the `TorchConnector` facilitates its use of quantum computed networks.

* **Sparse**, may be installed using command `pip install 'qiskit-machine-learning[sparse]'` to install the
  package. Sparse being installed will enable the usage of sparse arrays/tensors.

----

Ready to get going?...
======================

.. raw:: html

   <div class="tutorials-callout-container">
      <div class="row">

.. customcalloutitem::
   :description: Find out about Qiskit Machine Learning.
   :header: Dive into the tutorials
   :button_link:  ./tutorials/index.html
   :button_text: Qiskit Machine Learning tutorials

.. raw:: html

      </div>
   </div>


.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
