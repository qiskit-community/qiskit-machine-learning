# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gradients (:mod:`qiskit_machine_learning.gradients`)
==============================================
Algorithms to calculate the gradient of a quantum circuit.

.. currentmodule:: qiskit_machine_learning.gradients

Base Classes
------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseEstimatorGradient
   BaseQGT
   BaseSamplerGradient
   EstimatorGradientResult
   SamplerGradientResult
   QGTResult

Linear Combination of Unitaries
-------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinCombEstimatorGradient
   LinCombSamplerGradient
   LinCombQGT

Parameter Shift Rules
---------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ParamShiftEstimatorGradient
   ParamShiftSamplerGradient

Quantum Fisher Information
--------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QFIResult
   QFI

Simultaneous Perturbation Stochastic Approximation
--------------------------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SPSAEstimatorGradient
   SPSASamplerGradient
"""

from .base.base_estimator_gradient import BaseEstimatorGradient
from .base.base_qgt import BaseQGT
from .base.base_sampler_gradient import BaseSamplerGradient
from .base.estimator_gradient_result import EstimatorGradientResult
from .lin_comb.lin_comb_estimator_gradient import DerivativeType, LinCombEstimatorGradient
from .lin_comb.lin_comb_qgt import LinCombQGT
from .lin_comb.lin_comb_sampler_gradient import LinCombSamplerGradient
from .param_shift.param_shift_estimator_gradient import ParamShiftEstimatorGradient
from .param_shift.param_shift_sampler_gradient import ParamShiftSamplerGradient
from .qfi import QFI
from .qfi_result import QFIResult
from .base.qgt_result import QGTResult
from .base.sampler_gradient_result import SamplerGradientResult
from .spsa.spsa_estimator_gradient import SPSAEstimatorGradient
from .spsa.spsa_sampler_gradient import SPSASamplerGradient

__all__ = [
    "BaseEstimatorGradient",
    "BaseQGT",
    "BaseSamplerGradient",
    "DerivativeType",
    "EstimatorGradientResult",
    "LinCombEstimatorGradient",
    "LinCombQGT",
    "LinCombSamplerGradient",
    "ParamShiftEstimatorGradient",
    "ParamShiftSamplerGradient",
    "QFI",
    "QFIResult",
    "QGTResult",
    "SamplerGradientResult",
    "SPSAEstimatorGradient",
    "SPSASamplerGradient",
]
