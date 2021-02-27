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

"""Abstract base class for the binary classifier and the multiclass classifier."""

from abc import ABC, abstractmethod

# pylint: disable=invalid-name


class _QSVM_ABC(ABC):
    """Abstract base class for the binary classifier and the multiclass classifier."""

    def __init__(self, qalgo):

        self._qalgo = qalgo
        self._ret = {}

    @abstractmethod
    def run(self):
        """ run """
        raise NotImplementedError("Must have implemented this.")

    @property
    def ret(self):
        """ return result """
        return self._ret

    @ret.setter
    def ret(self, new_ret):
        """ sets result """
        self._ret = new_ret
