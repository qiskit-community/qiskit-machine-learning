# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===================================================
Qiskit's Machine Learning module (:mod:`qiskit.ml`)
===================================================

.. currentmodule:: qiskit.ml

This is the Qiskit`s machine learning module. There is an initial set of function here that
will be built out over time. At present it has sample sets that can be used with
Aqua's :mod:`~qiskit.aqua.algorithms.classifiers`.

Submodules
==========

.. autosummary::
   :toctree:

   datasets

"""

from ._logging import (get_qiskit_ml_logging,
                       set_qiskit_ml_logging)

__all__ = ['get_qiskit_ml_logging',
           'set_qiskit_ml_logging']
