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
   QSVM
   VQC
   SklearnSVM

Distribution Learners
+++++++++++++++++++++

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QGAN

Regressors
+++++++++++
Algorithms for data regression.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QSVR

"""

from .classifiers import QSVC, VQC, QSVM, SklearnSVM
from .distribution_learners import QGAN
from .regressors import QSVR

__all__ = [
    'QSVC',
    'VQC',
    'QSVM',
    'SklearnSVM',
    'QGAN',
    'QSVR'
]
