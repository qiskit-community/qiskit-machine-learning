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

r"""
Gradients (:mod:`qiskit_machine_learning.gradients`)
====================================================

Algorithms to calculate the gradient of a cost landscape to optimize a given objective function.

.. currentmodule:: qiskit_machine_learning.gradients

Base Classes
------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseEstimatorGradient
   BaseSamplerGradient
   EstimatorGradientResult
   SamplerGradientResult

Linear combination of unitaries
-------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinCombEstimatorGradient
   LinCombSamplerGradient

Parameter-shift rules
---------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ParamShiftEstimatorGradient
   ParamShiftSamplerGradient

Simultaneous perturbation stochastic approximation
--------------------------------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   SPSAEstimatorGradient
   SPSASamplerGradient
"""

from .base.base_estimator_gradient import BaseEstimatorGradient
from .base.base_sampler_gradient import BaseSamplerGradient
from .base.estimator_gradient_result import EstimatorGradientResult
from .lin_comb.lin_comb_estimator_gradient import DerivativeType, LinCombEstimatorGradient
from .lin_comb.lin_comb_sampler_gradient import LinCombSamplerGradient
from .param_shift.param_shift_estimator_gradient import ParamShiftEstimatorGradient
from .param_shift.param_shift_sampler_gradient import ParamShiftSamplerGradient
from .base.sampler_gradient_result import SamplerGradientResult
from .spsa.spsa_estimator_gradient import SPSAEstimatorGradient
from .spsa.spsa_sampler_gradient import SPSASamplerGradient

__all__ = [
    "BaseEstimatorGradient",
    "BaseSamplerGradient",
    "DerivativeType",
    "EstimatorGradientResult",
    "LinCombEstimatorGradient",
    "LinCombSamplerGradient",
    "ParamShiftEstimatorGradient",
    "ParamShiftSamplerGradient",
    "SamplerGradientResult",
    "SPSAEstimatorGradient",
    "SPSASamplerGradient",
]
