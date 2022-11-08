# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utility functions and classes (:mod:`qiskit_machine_learning.utils`)
====================================================================

A collection of utility functions and classes provided by Qiskit Machine Learning.

.. currentmodule:: qiskit_machine_learning.utils

Utilities
==========

.. autosummary::
   :toctree:

   loss_functions

"""
from .adjust_num_qubits import derive_num_qubits_feature_map_ansatz

__all__ = ["derive_num_qubits_feature_map_ansatz"]
