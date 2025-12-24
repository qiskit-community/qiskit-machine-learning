:orphan:

###############
Getting started
###############

Installation
============

Qiskit Machine Learning depends on Qiskit, which has its own
`installation instructions <https://quantum.cloud.ibm.com/docs/guides/install-qiskit>`__ detailing
installation options and its supported environments/platforms. You should refer to
that first. Then the information here can be followed which focuses on the additional installation
specific to Qiskit Machine Learning.

Qiskit Machine Learning has some functions that have been made optional where the dependent code and/or
support program(s) are not (or cannot be) installed by default. Those are PyTorch, Sparse and NLopt.
See :ref:`optional_installs` for more information.

.. tab-set::

    .. tab-item:: Start locally

        The simplest way to get started is to follow the installation guide for Qiskit `here <https://quantum.cloud.ibm.com/docs/guides/install-qiskit>`__

        In your virtual environment, where you installed Qiskit, install ``qiskit-machine-learning`` as follows:

        .. code:: sh

            pip install qiskit-machine-learning

        .. note::

            As Qiskit Machine Learning depends on Qiskit, you can though simply install it into your
            environment, as above, and pip will automatically install a compatible version of Qiskit
            if one is not already installed.

    .. tab-item:: Install from source

       Installing Qiskit Machine Learning from source allows you to access the most recently
       updated version under development instead of using the version in the Python Package
       Index (PyPI) repository. This will give you the ability to inspect and extend
       the latest version of the Qiskit Machine Learning code more efficiently.

       Since Qiskit Machine Learning depends on Qiskit, and its latest changes may require new or changed
       features of Qiskit, you should first follow Qiskit's `"Install from source"` instructions
       `here <https://quantum.cloud.ibm.com/docs/guides/install-qiskit-source>`__

       .. raw:: html

          <h2>Installing Qiskit Machine Learning from Source</h2>

       Using the same development environment that you installed Qiskit in you are ready to install
       Qiskit Machine Learning.

       1. Clone the Qiskit Machine Learning repository.

          .. code:: sh

             git clone https://github.com/qiskit-community/qiskit-machine-learning.git

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

* **PyTorch**, may be installed either using command ``pip install 'qiskit-machine-learning[torch]'`` to install the
  package or refer to PyTorch `getting started <https://pytorch.org/get-started/locally/>`__. When PyTorch
  is installed, the `TorchConnector` facilitates its use of quantum computed networks.

* **Sparse**, may be installed using command ``pip install 'qiskit-machine-learning[sparse]'`` to install the
  package. Sparse being installed will enable the usage of sparse arrays/tensors.

* **NLopt** is required for the global optimizers. `NLOpt <https://nlopt.readthedocs.io/en/latest/>`__
  can be installed manually with ``pip install nlopt`` on Windows and Linux platforms, or with
  ``brew install nlopt`` on MacOS using the Homebrew package manager. For more information, refer
  to the `installation guide <https://nlopt.readthedocs.io/en/latest/NLopt_Installation/>`__.


.. _contributing:

How can I contribute?
=====================

If you'd like to contribute to Qiskit, please take a look at our
`contribution guidelines <https://github.com/qiskit-community/qiskit-machine-learning/blob/main/CONTRIBUTING.md>`_.
This project adheres to the Qiskit `code of conduct <https://github.com/qiskit-community/qiskit-machine-learning/blob/main/CODE_OF_CONDUCT.md>`_.
By participating, you are expected to uphold this code.

We use `GitHub issues <https://github.com/qiskit-community/qiskit-machine-learning/issues>`_ for tracking requests and bugs. Please
`join the Qiskit Slack community <https://qisk.it/join-slack>`_
and use the `#qiskit-machine-learning <https://qiskit.enterprise.slack.com/archives/C07JE3V55C1>`_ 
channel for discussions and short questions.
For questions that are more suited for a forum, you can use the **Qiskit** tag in `Stack Overflow <https://stackoverflow.com/questions/tagged/qiskit>`_.

.. _citing:

How can I cite Qiskit Machine Learning?
=======================================

If you use Qiskit Machine Learning in your work, please cite the "overview" `ArXiv paper <https://arxiv.org/abs/2505.17756>`_ to 
support the continued development and visibility of the library.


----

Ready to get going?...
======================

.. raw:: html

   <div class="tutorials-callout-container">
      <div class="row">

.. qiskit-call-to-action-item::
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
