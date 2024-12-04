# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum machine learning algorithms (:mod:`qiskit_machine_learning.algorithms`)
===============================================================================

The package contains core algorithms such as classifiers and classifiers.

.. currentmodule:: qiskit_machine_learning.algorithms

Machine Learning Base Classes
-----------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TrainableModel
   ObjectiveFunction
   SerializableModelMixin

Machine Learning Objective Functions
------------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BinaryObjectiveFunction
   MultiClassObjectiveFunction
   OneHotObjectiveFunction

Algorithms
----------

Classifiers
+++++++++++
Algorithms for data classification.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PegasosQSVC
   QSVC
   VQC
   NeuralNetworkClassifier

Regressors
++++++++++
Algorithms for data regression.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QSVR
   VQR
   NeuralNetworkRegressor

Inference
+++++++++++
Algorithms for inference.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QBayesian
"""
from .trainable_model import TrainableModel
from .serializable_model import SerializableModelMixin
from .objective_functions import (
    ObjectiveFunction,
    BinaryObjectiveFunction,
    MultiClassObjectiveFunction,
    OneHotObjectiveFunction,
)
from .classifiers import QSVC, PegasosQSVC, VQC, NeuralNetworkClassifier
from .inference import QBayesian
from .regressors import QSVR, VQR, NeuralNetworkRegressor

__all__ = [
    "TrainableModel",
    "SerializableModelMixin",
    "ObjectiveFunction",
    "BinaryObjectiveFunction",
    "MultiClassObjectiveFunction",
    "OneHotObjectiveFunction",
    "QSVC",
    "PegasosQSVC",
    "NeuralNetworkClassifier",
    "VQC",
    "QSVR",
    "NeuralNetworkRegressor",
    "VQR",
    "QBayesian",
]
