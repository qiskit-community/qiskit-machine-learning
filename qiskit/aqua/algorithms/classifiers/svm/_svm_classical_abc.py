# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
abstract base class for the binary classifier and the multiclass classifier
"""

from abc import ABC, abstractmethod

from qiskit.aqua.utils import split_dataset_to_data_and_labels

# pylint: disable=invalid-name


class _SVM_Classical_ABC(ABC):
    """
    abstract base class for the binary classifier and the multiclass classifier
    """

    def __init__(self, training_dataset, test_dataset=None, datapoints=None, gamma=None):
        if training_dataset is None:
            raise ValueError('training dataset is missing! please provide it')

        self.training_dataset, self.class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        if test_dataset is not None:
            self.test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                 self.class_to_label)

        self.label_to_class = {label: class_name for class_name, label
                               in self.class_to_label.items()}
        self.num_classes = len(list(self.class_to_label.keys()))

        self.datapoints = datapoints
        self.gamma = gamma
        self._ret = {}

    @abstractmethod
    def run(self):
        """ run """
        raise NotImplementedError("Should have implemented this")

    @property
    def ret(self):
        """ return result """
        return self._ret

    @ret.setter
    def ret(self, new_ret):
        """ sets result """
        self._ret = new_ret
