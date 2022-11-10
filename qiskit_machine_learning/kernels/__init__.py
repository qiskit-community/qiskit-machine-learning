# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum kernels (:mod:`qiskit_machine_learning.kernels`)
========================================================

A set of extendable classes that can be used to evaluate kernel matrices.

.. currentmodule:: qiskit_machine_learning.kernels

Quantum Kernels
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QuantumKernel
   BaseKernel
   FidelityQuantumKernel
   TrainableKernel
   TrainableFidelityQuantumKernel

Submodules
==========

.. autosummary::
   :toctree:

   algorithms
"""

from .quantum_kernel import QuantumKernel
from .base_kernel import BaseKernel
from .fidelity_quantum_kernel import FidelityQuantumKernel
from .trainable_kernel import TrainableKernel
from .trainable_fidelity_quantum_kernel import TrainableFidelityQuantumKernel

__all__ = [
    "QuantumKernel",
    "BaseKernel",
    "FidelityQuantumKernel",
    "TrainableKernel",
    "TrainableFidelityQuantumKernel",
]
