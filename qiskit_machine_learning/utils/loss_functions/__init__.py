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
Loss Functions (:mod:`qiskit_machine_learning.utils.loss_functions`)
====================================================================

A collection of common loss functions to be used with the classifiers and regressors provided
by Qiskit Machine Learning.

.. currentmodule:: qiskit_machine_learning.utils.loss_functions

Loss Function Base Class
------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Loss
   KernelLoss

Loss Functions
--------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   L1Loss
   L2Loss
   CrossEntropyLoss
   SVCLoss
"""

from .loss_functions import (
    Loss,
    L1Loss,
    L2Loss,
    CrossEntropyLoss,
)

from .kernel_loss_functions import KernelLoss, SVCLoss

__all__ = [
    "Loss",
    "KernelLoss",
    "L1Loss",
    "L2Loss",
    "CrossEntropyLoss",
    "SVCLoss",
]
