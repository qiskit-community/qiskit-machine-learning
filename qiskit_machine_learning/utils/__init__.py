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
Utilities (:mod:`qiskit_machine_learning.utils`)
================================================
Various utility functionality...

.. currentmodule:: qiskit_machine_learning.utils

Utilities
=========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   get_feature_dimension
   get_num_classes
   split_dataset_to_data_and_labels
   map_label_to_class_name
   reduce_dim_to_via_pca
   optimize_svm

"""

from .dataset_helper import (get_feature_dimension, get_num_classes,
                             split_dataset_to_data_and_labels,
                             map_label_to_class_name, reduce_dim_to_via_pca)
from .qp_solver import optimize_svm

__all__ = [
    'get_feature_dimension',
    'get_num_classes',
    'split_dataset_to_data_and_labels',
    'map_label_to_class_name',
    'reduce_dim_to_via_pca',
    'optimize_svm',
]
