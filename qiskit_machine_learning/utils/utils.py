# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Qiskit machine learning utils"""

import numpy as np


def make_2d(array: np.ndarray, n_copies: int):
    """
    Takes a 1D numpy array and copies n times it along a second axis.
    """
    return np.repeat(array[np.newaxis, :], n_copies, axis=0)
