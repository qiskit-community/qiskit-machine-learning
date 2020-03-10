# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Multi-class Extensions (:mod:`qiskit.aqua.components.multiclass_extensions`)
============================================================================
A multiclass extension allows Aqua's binary classifier algorithms, such as
:class:`~qiskit.aqua.algorithms.QSVM` and :class:`~qiskit.aqua.algorithms.SklearnSVM`
to handle more than two classes and do
`multiclass classification <https://en.wikipedia.org/wiki/Multiclass_classification>`_.

The multiclass extensions use different techniques to perform multiclass classification
using the underlying binary classifier.

.. currentmodule:: qiskit.aqua.components.multiclass_extensions

Multiclass Extension Base Class
===============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   MulticlassExtension

Multiclass Extensions
=====================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AllPairs
   OneAgainstRest
   ErrorCorrectingCode

"""

from .estimator import Estimator
from .multiclass_extension import MulticlassExtension
from .all_pairs import AllPairs
from .one_against_rest import OneAgainstRest
from .error_correcting_code import ErrorCorrectingCode

__all__ = ['Estimator',
           'MulticlassExtension',
           'AllPairs',
           'OneAgainstRest',
           'ErrorCorrectingCode']
