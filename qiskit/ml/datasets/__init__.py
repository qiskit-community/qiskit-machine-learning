# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Datasets (:mod:`qiskit.ml.datasets`)
===================================================
Sample datasets suitable for machine learning problems

.. currentmodule:: qiskit.ml.datasets

Datasets
========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ad_hoc_data
   sample_ad_hoc_data
   breast_cancer
   digits
   gaussian
   iris
   wine

"""

from .ad_hoc import ad_hoc_data, sample_ad_hoc_data
from .breast_cancer import breast_cancer
from .digits import digits
from .gaussian import gaussian
from .iris import iris
from .wine import wine

__all__ = ['ad_hoc_data',
           'sample_ad_hoc_data',
           'breast_cancer',
           'digits',
           'gaussian',
           'iris',
           'wine']
