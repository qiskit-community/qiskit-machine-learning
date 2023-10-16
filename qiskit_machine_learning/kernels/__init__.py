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

r"""
Quantum kernels (:mod:`qiskit_machine_learning.kernels`)
========================================================

A set of extendable classes that can be used to evaluate kernel matrices.

The general task of machine learning is to find and study patterns in data. For many
algorithms, the datapoints are better understood in a higher dimensional feature space,
through the use of a kernel function:

.. math::

    K(x, y) = \langle f(x), f(y)\rangle.

Here :math:`K` is the kernel function, :math:`x`, :math:`y` are :math:`n` dimensional inputs.
:math:`f` is a map from :math:`n`-dimension to :math:`m`-dimension space. :math:`\langle x, y
\rangle` denotes the inner product. Usually :math:`m` is much larger than :math:`n`.

The quantum kernel algorithm calculates a kernel matrix, given datapoints :math:`x` and
:math:`y` and feature map :math:`f`, all of :math:`n` dimension. This kernel matrix can then be
used in classical machine learning algorithms such as support vector classification, spectral
clustering or ridge regression.

.. currentmodule:: qiskit_machine_learning.kernels

Quantum Kernels
---------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseKernel
   FidelityQuantumKernel
   FidelityStatevectorKernel
   TrainableKernel
   TrainableFidelityQuantumKernel
   TrainableFidelityStatevectorKernel

Submodules
----------

.. autosummary::
   :toctree:

   algorithms
"""

from .base_kernel import BaseKernel
from .fidelity_quantum_kernel import FidelityQuantumKernel
from .fidelity_statevector_kernel import FidelityStatevectorKernel
from .trainable_kernel import TrainableKernel
from .trainable_fidelity_quantum_kernel import TrainableFidelityQuantumKernel
from .trainable_fidelity_statevector_kernel import TrainableFidelityStatevectorKernel

__all__ = [
    "BaseKernel",
    "FidelityQuantumKernel",
    "FidelityStatevectorKernel",
    "TrainableKernel",
    "TrainableFidelityQuantumKernel",
    "TrainableFidelityStatevectorKernel",
]
