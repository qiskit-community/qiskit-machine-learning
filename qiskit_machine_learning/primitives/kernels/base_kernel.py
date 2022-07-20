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

"""Base kernel"""

from abc import abstractmethod
from typing import Tuple

import numpy as np


class BaseKernel(ABC):
    r"""
    Abstract class defining the Kernel interface.

    The general task of machine learning is to find and study patterns in data. For many
    algorithms, the datapoints are better understood in a higher dimensional feature space,
    through the use of a kernel function:

    .. math::

        K(x, y) = \langle f(x), f(y)\rangle.

    Here K is the kernel function, x, y are n dimensional inputs. f is a map from n-dimension
    to m-dimension space. :math:`\langle x, y \rangle` denotes the dot product.
    Usually m is much larger than n.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints x and y and feature
    map f, all of n dimension. This kernel matrix can then be used in classical machine learning
    algorithms such as support vector classification, spectral clustering or ridge regression.
    """

    def __init__(self, enforce_psd: bool = True) -> None:
        """
        Args:
            enforce_psd: Project to closest positive semidefinite matrix if x = y.
                Default True.
        """
        self._num_features: int = 0
        self._enforce_psd = enforce_psd

    @abstractmethod
    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        r"""
        Construct kernel matrix for given data.

        If y_vec is None, self inner product is calculated.

        Args:
            x_vec: 1D or 2D array of datapoints, NxD, where N is the number of datapoints,
                                                            D is the feature dimension
            y_vec: 1D or 2D array of datapoints, MxD, where M is the number of datapoints,
                                                            D is the feature dimension

        Returns:
            2D matrix, NxM

        Raises:
            QiskitMachineLearningError:
                A quantum instance or backend has not been provided
            ValueError:
                x_vec and/or y_vec are not one or two dimensional arrays
                x_vec and y_vec have have incompatible dimensions
                x_vec and/or y_vec have incompatible dimension with feature map and
                    and feature map can not be modified to match.
        """
        raise NotImplementedError()

    def _check_and_reshape(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> Tuple[np.ndarray]:
        r"""
        Performs checks on the dimensions of the input data x_vec and y_vec.
        Reshapes the arrays so that `x_vec.shape = (N,D)` and `y_vec.shape = (M,D)`.
        """
        if not isinstance(x_vec, np.ndarray):
            x_vec = np.asarray(x_vec)

        if y_vec is not None and not isinstance(y_vec, np.ndarray):
            y_vec = np.asarray(y_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = x_vec.reshape(1, -1)

        if y_vec is not None and y_vec.ndim > 2:
            raise ValueError("y_vec must be a 1D or 2D array")

        if y_vec is not None and y_vec.ndim == 1:
            y_vec = y_vec.reshape(1, -1)

        if y_vec is not None and y_vec.shape[1] != x_vec.shape[1]:
            raise ValueError(
                "x_vec and y_vec have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
            )

        if x_vec.shape[1] != self._num_features:
            raise ValueError(
                "x_vec and class feature map have incompatible dimensions.\n"
                f"x_vec has {x_vec.shape[1]} dimensions, "
                f"but feature map has {self._num_features}."
            )

        if y_vec is None:
            y_vec = x_vec

        return x_vec, y_vec

    # pylint: disable=invalid-name
    def _make_psd(self, kernel_matrix: np.ndarray) -> np.ndarray:
        r"""
        Find the closest positive semi-definite approximation to symmetric kernel matrix.
        The (symmetric) matrix should always be positive semi-definite by construction,
        but this can be violated in case of noise, such as sampling noise, thus the
        adjustment is only done if NOT using the statevector simulation.

        Args:
            kernel_matrix: symmetric 2D array of the kernel entries
        """
        d, u = np.linalg.eig(kernel_matrix)
        return u @ np.diag(np.maximum(0, d)) @ u.transpose()
