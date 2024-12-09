# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum neural networks (:mod:`qiskit_machine_learning.neural_networks`)
========================================================================

A neural network is a parametrized network which may be defined as a artificial
neural network - classical neural network - or as parametrized quantum circuits
- quantum neural network. Furthermore, neural networks can be defined with respect
to a discriminative or generative task.

Neural networks may be used, for example, with the
:class:`~qiskit_machine_learning.algorithms.VQC` algorithm.

See also the :class:`~qiskit_machine_learning.connectors.TorchConnector` that allows the
use of these neural networks in code written to `PyTorch <https://pytorch.org/>`_.

.. currentmodule:: qiskit_machine_learning.neural_networks

Neural Network Base Classes
---------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NeuralNetwork

Neural networks
---------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EstimatorQNN
   SamplerQNN

Metrics for neural networks
---------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   EffectiveDimension
   LocalEffectiveDimension
"""

from .effective_dimension import EffectiveDimension, LocalEffectiveDimension
from .estimator_qnn import EstimatorQNN
from .neural_network import NeuralNetwork
from .sampler_qnn import SamplerQNN

__all__ = [
    "NeuralNetwork",
    "EffectiveDimension",
    "LocalEffectiveDimension",
    "EstimatorQNN",
    "SamplerQNN",
]
