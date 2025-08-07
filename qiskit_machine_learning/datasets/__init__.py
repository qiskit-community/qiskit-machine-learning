# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2025.
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

A collection of synthetic datasets used to test and benchmark machine-learning
algorithms implemented in Qiskit Machine Learning.

.. currentmodule:: qiskit_machine_learning.datasets

Synthetic dataset generators
----------------------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ad_hoc_data
   entanglement_concentration_data
"""

from .ad_hoc import ad_hoc_data
from .entanglement_concentration import entanglement_concentration_data

__all__ = ["ad_hoc_data", "entanglement_concentration_data"]
