#####################################
Qiskit Machine Learning overview
#####################################

Overview
==============

Qiskit Machine Learning introduces fundamental computational building blocks - such as Quantum Kernels
and Quantum Neural Networks - used in different applications, including classification and regression.
On the one hand, this design is very easy to use and allows users to rapidly prototype a first model
without deep quantum computing knowledge. On the other hand, Qiskit Machine Learning is very flexible,
and users can easily extend it to support cutting-edge quantum machine learning research.

Qiskit Machine Learning provides the :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`
class class that makes use of the :class:`~qiskit_algorithms.state_fidelities.BaseStateFidelity` algorithm
introduced in Qiskit and can be easily used to directly compute kernel matrices for given datasets
or can be passed to a Quantum Support Vector Classifier
(:class:`~qiskit_machine_learning.algorithms.QSVC`) or
Quantum Support Vector Regressor (:class:`~qiskit_machine_learning.algorithms.QSVR`)
to quickly start solving classification or regression problems.
It also can be used with many other existing kernel-based machine learning algorithms from established
classical frameworks.

Qiskit Machine Learning defines a generic interface for neural networks that is implemented by different
quantum neural networks. Two core implementations are readily provided, such as the
:class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`
and the :class:`~qiskit_machine_learning.neural_networks.SamplerQNN`.
The :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN` leverages
the :class:`~qiskit.primitives.BaseEstimator` primitive from Qiskit and allows users to combine
parametrized quantum circuits with quantum mechanical observables. The circuits can be constructed
using, for example, building blocks from Qiskit's circuit library, and the QNN's output is given
by the expected value of the observable.
The :class:`~qiskit_machine_learning.neural_networks.SamplerQNN` leverages another primitive
introduced in Qiskit, the :class:`~qiskit.primitives.BaseSampler` primitive. This neural network
translates quasi-probabilities of bitstrings estimated by the primitive into a desired output. This
translation step can be used to interpret a given bitstring in a particular context, e.g.
translating it into a set of classes.

The neural networks include the functionality to evaluate them for a given input as well as to compute the
corresponding gradients, which is important for efficient training. To train and use neural networks,
Qiskit Machine Learning provides a variety of learning algorithms such as the
:class:`~qiskit_machine_learning.algorithms.NeuralNetworkClassifier` and
:class:`~qiskit_machine_learning.algorithms.NeuralNetworkRegressor`.
Both take a QNN as input and then use it in a classification or regression context.
To allow an easy start, two convenience implementations are provided - the Variational Quantum Classifier
(:class:`~qiskit_machine_learning.algorithms.VQC`)
as well as the Variational Quantum Regressor (:class:`~qiskit_machine_learning.algorithms.VQR`).
Both take just a feature map and an ansatz and construct the underlying QNN automatically.

In addition to the models provided directly in Qiskit Machine Learning, it has the
:class:`~qiskit_machine_learning.connectors.TorchConnector`,
which allows users to integrate all of our quantum neural networks directly into the
`PyTorch <https://pytorch.org>`__
open source machine learning library. Thanks to Qiskit Algorithm's gradient algorithms,
this includes automatic
differentiation - the overall gradients computed by `PyTorch <https://pytorch.org>`__
during the backpropagation take into
account quantum neural networks, too. The flexible design also allows the building of connectors
to other packages in the future.



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

