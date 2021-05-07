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
Loss Functions (:mod:`qiskit_machine_learning.utils.loss_functions`)
====================================================================
A collections of common loss functions to be used with the classifiers and regressors provided
by Qiskit Machine Learning.

.. currentmodule:: qiskit_machine_learning.utils.loss_functions

Loss Function Base Class
========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Loss

Loss Functions
==============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   L1Loss
   L2Loss
   CrossEntropyLoss
   CrossEntropySigmoidLoss

"""

from .loss_functions import (
    Loss,
    L1Loss,
    L2Loss,
    CrossEntropyLoss,
    CrossEntropySigmoidLoss,
)

__all__ = ["Loss", "L1Loss", "L2Loss", "CrossEntropyLoss", "CrossEntropySigmoidLoss"]
