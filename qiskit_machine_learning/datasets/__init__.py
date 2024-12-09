# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Datasets (:mod:`qiskit_machine_learning.datasets`)
==================================================

A set of sample datasets to test machine learning algorithms.

.. currentmodule:: qiskit_machine_learning.datasets

Datasets
--------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ad_hoc_data

"""

from .ad_hoc import ad_hoc_data

__all__ = [
    "ad_hoc_data",
]
