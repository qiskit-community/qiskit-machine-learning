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

Machine Learning Base Classes
=============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TrainableModel
   ObjectiveFunction

Machine Learning Objective Functions
====================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BinaryObjectiveFunction
   MultiClassObjectiveFunction
   OneHotObjectiveFunction

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

from .trainable_model import TrainableModel
from .objective_functions import (
    ObjectiveFunction,
    BinaryObjectiveFunction,
    MultiClassObjectiveFunction,
    OneHotObjectiveFunction,
)
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
    "TrainableModel",
    "ObjectiveFunction",
    "BinaryObjectiveFunction",
    "MultiClassObjectiveFunction",
    "OneHotObjectiveFunction",
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
