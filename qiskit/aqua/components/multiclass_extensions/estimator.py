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

""" estimator """

from abc import ABC, abstractmethod


class Estimator(ABC):
    """ Estimator class """
    @abstractmethod
    def fit(self, x, y):
        """ fit """
        raise NotImplementedError("Should have implemented this")

    @abstractmethod
    def decision_function(self, x):
        """ decision function """
        raise NotImplementedError("Should have implemented this")
