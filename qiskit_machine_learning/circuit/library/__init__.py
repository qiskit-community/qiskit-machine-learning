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

"""
Circuit library for Machine Learning applications (:mod:`qiskit_machine_learning.circuit.library`)
==================================================================================================

A library of quantum circuits used for machine learning applications and as
building blocks for machine learning algorithms.

.. currentmodule:: qiskit_machine_learning.circuit.library

Feature Maps
============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   RawFeatureVector

"""

from .raw_feature_vector import RawFeatureVector

__all__ = [
    "RawFeatureVector",
]
