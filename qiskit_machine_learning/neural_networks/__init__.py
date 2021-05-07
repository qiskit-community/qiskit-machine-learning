# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Neural Networks (:mod:`qiskit_machine_learning.neural_networks`)
================================================================
A neural network is a parametrized network which may be defined as a artificial
neural network - classical neural network - or as parametrized quantum circuits
- quantum neural network. Furthermore, neural networks can be defined with respect
to a discriminative or generative task.

Neural Networks may be used, for example, with the
:class:`~qiskit_machine_learning.algorithms.QGAN` algorithm.

.. currentmodule:: qiskit_machine_learning.neural_networks

Neural Network Base Classes
===========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NeuralNetwork

Neural Networks
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NeuralNetwork
   OpflowQNN
   TwoLayerQNN
   SamplingNeuralNetwork
   CircuitQNN

"""

from .circuit_qnn import CircuitQNN
from .neural_network import NeuralNetwork
from .opflow_qnn import OpflowQNN
from .sampling_neural_network import SamplingNeuralNetwork
from .two_layer_qnn import TwoLayerQNN

__all__ = [
    "NeuralNetwork",
    "OpflowQNN",
    "TwoLayerQNN",
    "SamplingNeuralNetwork",
    "CircuitQNN",
]
