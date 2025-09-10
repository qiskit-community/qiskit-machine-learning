# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base kernel"""

from __future__ import annotations

from abc import abstractmethod, ABC

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap


class BaseKernel(ABC):
    r"""
    An abstract definition of the quantum kernel interface.

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

    def __init__(self, *, feature_map: QuantumCircuit = None, enforce_psd: bool = True) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
            enforce_psd: Project to closest positive semidefinite matrix if ``x = y``.
                Default ``True``.
        """
        if feature_map is None:
            feature_map = ZZFeatureMap(2)

        self._num_features = feature_map.num_parameters
        self._feature_map = feature_map
        self._enforce_psd = enforce_psd

    @abstractmethod
    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
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
        """
        raise NotImplementedError()

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the feature map of this kernel."""
        return self._feature_map

    @property
    def num_features(self) -> int:
        """Returns the number of features in this kernel."""
        return self._num_features

    @property
    def enforce_psd(self) -> bool:
        """
        Returns ``True`` if the kernel matrix is required to project to the closest positive
        semidefinite matrix.
        """
        return self._enforce_psd

    def _validate_input(
        self, x_vec: np.ndarray, y_vec: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        x_vec = np.asarray(x_vec)

        if x_vec.ndim > 2:
            raise ValueError("x_vec must be a 1D or 2D array")

        if x_vec.ndim == 1:
            x_vec = np.reshape(x_vec, (-1, len(x_vec)))

        if x_vec.shape[1] != self._num_features:
            # before raising an error we try to adjust the feature map
            # to the required number of qubit.
            try:
                self._feature_map.num_qubits = x_vec.shape[1]
            except AttributeError as a_e:
                raise ValueError(
                    f"x_vec and class feature map have incompatible dimensions.\n"
                    f"x_vec has {x_vec.shape[1]} dimensions, "
                    f"but feature map has {self._num_features}."
                ) from a_e

        if y_vec is not None:
            y_vec = np.asarray(y_vec)

            if y_vec.ndim == 1:
                y_vec = np.reshape(y_vec, (-1, len(y_vec)))

            if y_vec.ndim > 2:
                raise ValueError("y_vec must be a 1D or 2D array")

            if y_vec.shape[1] != x_vec.shape[1]:
                raise ValueError(
                    "x_vec and y_vec have incompatible dimensions.\n"
                    f"x_vec has {x_vec.shape[1]} dimensions, but y_vec has {y_vec.shape[1]}."
                )

        return x_vec, y_vec

    def _make_psd(self, kernel_matrix: np.ndarray) -> np.ndarray:
        r"""
        Find the closest positive semi-definite approximation to a symmetric kernel matrix.
        The (symmetric) matrix should always be positive semi-definite by construction,
        but this can be violated in case of noise, such as sampling noise.

        Args:
            kernel_matrix: Symmetric 2D array of the kernel entries.

        Returns:
            The closest positive semi-definite matrix.
        """
        w, v = np.linalg.eig(kernel_matrix)
        m = v @ np.diag(np.maximum(0, w)) @ v.transpose()
        return m.real
