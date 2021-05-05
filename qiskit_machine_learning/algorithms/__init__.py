# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Algorithms (:mod:`qiskit_machine_learning.algorithms`)

.. currentmodule:: qiskit_machine_learning.algorithms

Algorithms
==========

Classifiers
+++++++++++
Algorithms for data classification.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QSVC
   NeuralNetworkClassifier
   VQC

Regressors
++++++++++
Quantum Support Vector Regressor.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QSVR
   NeuralNetworkRegressor
   VQR

Distribution Learners
+++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   DiscriminativeNetwork
   GenerativeNetwork
   NumPyDiscriminator
   PyTorchDiscriminator
   QuantumGenerator
   QGAN

"""

from .classifiers import QSVC, VQC, NeuralNetworkClassifier
from .regressors import QSVR, VQR, NeuralNetworkRegressor
from .distribution_learners import (
    DiscriminativeNetwork,
    GenerativeNetwork,
    NumPyDiscriminator,
    PyTorchDiscriminator,
    QuantumGenerator,
    QGAN,
)

__all__ = [
    "QSVC",
    "NeuralNetworkClassifier",
    "VQC",
    "QSVR",
    "NeuralNetworkRegressor",
    "VQR",
    "DiscriminativeNetwork",
    "GenerativeNetwork",
    "NumPyDiscriminator",
    "PyTorchDiscriminator",
    "QuantumGenerator",
    "QGAN",
]
