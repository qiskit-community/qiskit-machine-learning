# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Neural Networks (:mod:`qiskit.aqua.components.neural_networks`)
===============================================================
A neural network is a parametrized network which may be defined as a artificial
neural network - classical neural network - or as parametrized quantum circuits
- quantum neural network. Furthermore, neural networks can be defined with respect
to a discriminative or generative task.

Neural Networks may be used, for example, with the
:class:`~qiskit.aqua.algorithms.QGAN` algorithm.

.. currentmodule:: qiskit.aqua.components.neural_networks

Neural Network Base Classes
===========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   DiscriminativeNetwork
   GenerativeNetwork

Neural Networks
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   NumPyDiscriminator
   PyTorchDiscriminator
   QuantumGenerator

"""

from .generative_network import GenerativeNetwork
from .quantum_generator import QuantumGenerator
from .discriminative_network import DiscriminativeNetwork
from .numpy_discriminator import NumPyDiscriminator
from .pytorch_discriminator import PyTorchDiscriminator

__all__ = [
    'DiscriminativeNetwork',
    'GenerativeNetwork',
    'QuantumGenerator',
    'NumPyDiscriminator',
    'PyTorchDiscriminator'
]
