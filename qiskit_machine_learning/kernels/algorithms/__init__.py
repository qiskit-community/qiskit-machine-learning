# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum Kernel Algorithms (:mod:`qiskit_machine_learning.kernels.algorithms`)

.. currentmodule:: qiskit_machine_learning.kernels.algorithms

Quantum Kernel Algorithms
=========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QuantumKernelTrainer
   QuantumKernelTrainerResult

"""

from .quantum_kernel_trainer import QuantumKernelTrainer, QuantumKernelTrainerResult

__all__ = ["QuantumKernelTrainer", "QuantumKernelTrainerResult"]
