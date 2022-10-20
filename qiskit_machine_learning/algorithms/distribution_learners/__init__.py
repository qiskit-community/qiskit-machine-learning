# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Distribution Learners Package """

from .qgan import (
    DiscriminativeNetwork,
    GenerativeNetwork,
    NumPyDiscriminator,
    PyTorchDiscriminator,
    QuantumGenerator,
    QGAN,
)
from ...deprecation import warn_deprecated, DeprecatedType, MachineLearningDeprecationWarning

warn_deprecated(
    "0.5.0",
    old_type=DeprecatedType.PACKAGE,
    old_name="qiskit_machine_learning.algorithms.distribution_learners",
    new_type=DeprecatedType.PACKAGE,
    additional_msg="Please refer to the QGAN tutorial instead",
    stack_level=3,
    category=MachineLearningDeprecationWarning,
)

__all__ = [
    "DiscriminativeNetwork",
    "GenerativeNetwork",
    "NumPyDiscriminator",
    "PyTorchDiscriminator",
    "QuantumGenerator",
    "QGAN",
]
