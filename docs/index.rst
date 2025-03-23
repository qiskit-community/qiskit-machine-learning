#####################################
Qiskit Machine Learning overview
#####################################

Overview
==============

Qiskit Machine Learning introduces fundamental computational building blocks, such as Quantum
Kernels and Quantum Neural Networks, used in various applications including classification
and regression.

This library is part of the Qiskit Community ecosystem, a collection of high-level codes that are based
on the Qiskit software development kit. As of version ``0.7``, Qiskit Machine Learning is co-maintained
by IBM and the `Hartree Center <https://www.hartree.stfc.ac.uk/>`__, part of the UK Science and
Technologies Facilities Council (STFC).

The Qiskit Machine Learning framework aims to be:

* **User-friendly**, allowing users to quickly and easily prototype quantum machine learning models without
  the need of extensive quantum computing knowledge.
* **Flexible**, providing tools and functionalities to conduct proof-of-concepts and innovative research
  in quantum machine learning for both beginners and experts.
* **Extensible**, facilitating the integration of new cutting-edge features leveraging Qiskit's
  architectures, patterns and related services.

What are the main features of Qiskit Machine Learning?
======================================================

Kernel-based methods
---------------------

The :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`
class uses the :class:`~qiskit_algorithms.state_fidelities.BaseStateFidelity`
algorithm. It computes kernel matrices for datasets and can be combined with a Quantum Support Vector Classifier (:class:`~qiskit_machine_learning.algorithms.QSVC`)
or a Quantum Support Vector Regressor (:class:`~qiskit_machine_learning.algorithms.QSVR`)
to solve classification or regression problems respectively. It is also compatible with classical kernel-based machine learning algorithms.

Quantum Neural Networks (QNNs)
------------------------------

Qiskit Machine Learning defines a generic interface for neural networks, implemented by two core (derived) primitives:

- :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN` leverages the Qiskit
  `Estimator <https://docs.quantum.ibm.com/api/qiskit/qiskit.primitives.BaseEstimator>`__ primitive, combining parametrized quantum circuits
  with quantum mechanical observables. The output is the expected value of the observable.

- :class:`~qiskit_machine_learning.neural_networks.SamplerQNN` leverages the Qiskit
  `Sampler <https://docs.quantum.ibm.com/api/qiskit/qiskit.primitives.BaseSampler>`__ primitive,
  translating bit-string counts into the desired outputs.

To train and use neural networks, Qiskit Machine Learning provides learning algorithms such as the :class:`~qiskit_machine_learning.algorithms.NeuralNetworkClassifier`
and :class:`~qiskit_machine_learning.algorithms.NeuralNetworkRegressor`.
Finally, built on these, the Variational Quantum Classifier (:class:`~qiskit_machine_learning.algorithms.VQC`)
and the Variational Quantum Regressor (:class:`~qiskit_machine_learning.algorithms.VQR`)
take a *feature map* and an *ansatz* to construct the underlying QNN automatically using high-level syntax.

Integration with PyTorch
------------------------

The :class:`~qiskit_machine_learning.connectors.TorchConnector`
integrates QNNs with `PyTorch <https://pytorch.org>`_.
Thanks to the gradient algorithms in Qiskit Machine Learning, this includes automatic differentiation.
The overall gradients computed by PyTorch during the backpropagation take into account quantum neural
networks, too. The flexible design also allows the building of connectors to other packages in the future.




Next Steps
=================================

`Getting started <getting_started.html>`_

`Migration Guide <migration/index.html>`_

`Tutorials <tutorials/index.html>`_

.. toctree::
    :hidden:

    Overview <self>
    Getting Started <getting_started>
    Migration Guide <migration/index>
    Tutorials <tutorials/index>
    API Reference <apidocs/qiskit_machine_learning>
    Release Notes <release_notes>
    GitHub <https://github.com/qiskit-community/qiskit-machine-learning>



.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`

