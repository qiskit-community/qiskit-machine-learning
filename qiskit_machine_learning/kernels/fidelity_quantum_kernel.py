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
"""Fidelity Quantum Kernel"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.state_fidelities import BaseStateFidelity, ComputeUncompute
from qiskit.primitives import Sampler

from .base_kernel import BaseKernel

KernelIndices = List[Tuple[int, int]]


class FidelityQuantumKernel(BaseKernel):
    r"""
    An implementation of the quantum kernel interface based on the
    :class:`~qiskit.algorithms.state_fidelities.BaseStateFidelity` algorithm.

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

    Here, the kernel function is defined as the overlap of two quantum states defined by a
    parametrized quantum circuit (called feature map):

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        fidelity: BaseStateFidelity | None = None,
        enforce_psd: bool = True,
        evaluate_duplicates: str = "off_diagonal",
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
            fidelity: An instance of the
                :class:`~qiskit.algorithms.state_fidelities.BaseStateFidelity` primitive to be used
                to compute fidelity between states. Default is
                :class:`~qiskit.algorithms.state_fidelities.ComputeUncompute` which is created on
                top of the reference sampler defined by :class:`~qiskit.primitives.Sampler`.
            enforce_psd: Project to the closest positive semidefinite matrix if ``x = y``.
                Default ``True``.
            evaluate_duplicates: Defines a strategy how kernel matrix elements are evaluated if
               duplicate samples are found. Possible values are:

                    - ``all`` means that all kernel matrix elements are evaluated, even the diagonal
                      ones when training. This may introduce additional noise in the matrix.
                    - ``off_diagonal`` when training the matrix diagonal is set to `1`, the rest
                      elements are fully evaluated, e.g., for two identical samples in the
                      dataset. When inferring, all elements are evaluated. This is the default
                      value.
                    - ``none`` when training the diagonal is set to `1` and if two identical samples
                      are found in the dataset the corresponding matrix element is set to `1`.
                      When inferring, matrix elements for identical samples are set to `1`.
        Raises:
            ValueError: When unsupported value is passed to `evaluate_duplicates`.
        """
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)

        eval_duplicates = evaluate_duplicates.lower()
        if eval_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Unsupported value passed as evaluate_duplicates: {evaluate_duplicates}"
            )
        self._evaluate_duplicates = eval_duplicates

        if fidelity is None:
            fidelity = ComputeUncompute(sampler=Sampler())
        self._fidelity = fidelity

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        kernel_shape = (x_vec.shape[0], y_vec.shape[0])

        if is_symmetric:
            left_parameters, right_parameters, indices = self._get_symmetric_parameterization(x_vec)
            kernel_matrix = self._get_symmetric_kernel_matrix(
                kernel_shape, left_parameters, right_parameters, indices
            )
        else:
            left_parameters, right_parameters, indices = self._get_parameterization(x_vec, y_vec)
            kernel_matrix = self._get_kernel_matrix(
                kernel_shape, left_parameters, right_parameters, indices
            )

        if is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)

        # due to truncation and rounding errors we may get complex numbers
        kernel_matrix = np.real(kernel_matrix)

        return kernel_matrix

    def _get_parameterization(
        self, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, KernelIndices]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = []
        for i, x_i in enumerate(x_vec):
            for j, y_j in enumerate(y_vec):
                if self._is_trivial(i, j, x_i, y_j, False):
                    continue

                left_parameters = np.vstack((left_parameters, x_i))
                right_parameters = np.vstack((right_parameters, y_j))
                indices.append((i, j))

        return left_parameters, right_parameters, indices

    def _get_symmetric_parameterization(
        self, x_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, KernelIndices]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = []
        for i, x_i in enumerate(x_vec):
            for j, x_j in enumerate(x_vec[i:]):
                if self._is_trivial(i, i + j, x_i, x_j, True):
                    continue

                left_parameters = np.vstack((left_parameters, x_i))
                right_parameters = np.vstack((right_parameters, x_j))
                indices.append((i, i + j))

        return left_parameters, right_parameters, indices

    def _get_kernel_matrix(
        self,
        kernel_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: KernelIndices,
    ) -> np.ndarray:
        """
        Given a parameterization, this computes the symmetric kernel matrix.
        """
        kernel_entries = self._get_kernel_entries(left_parameters, right_parameters)

        # fill in trivial entries and then update with fidelity values
        kernel_matrix = np.ones(kernel_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]

        return kernel_matrix

    def _get_symmetric_kernel_matrix(
        self,
        kernel_shape: tuple[int, int],
        left_parameters: np.ndarray,
        right_parameters: np.ndarray,
        indices: KernelIndices,
    ) -> np.ndarray:
        """
        Given a set of parameterization, this computes the kernel matrix.
        """
        kernel_entries = self._get_kernel_entries(left_parameters, right_parameters)
        kernel_matrix = np.ones(kernel_shape)

        for i, (col, row) in enumerate(indices):
            kernel_matrix[col, row] = kernel_entries[i]
            kernel_matrix[row, col] = kernel_entries[i]

        return kernel_matrix

    def _get_kernel_entries(self, left_parameters: np.ndarray, right_parameters: np.ndarray):
        """
        Gets kernel entries by executing the underlying fidelity instance and getting the results
        back from the async job.
        """
        num_circuits = left_parameters.shape[0]
        if num_circuits != 0:
            job = self._fidelity.run(
                [self._feature_map] * num_circuits,
                [self._feature_map] * num_circuits,
                left_parameters,
                right_parameters,
            )
            kernel_entries = np.real(job.result().fidelities)
        else:
            # trivial case, only identical samples
            kernel_entries = []
        return kernel_entries

    def _is_trivial(
        self, i: int, j: int, x_i: np.ndarray, y_j: np.ndarray, symmetric: bool
    ) -> bool:
        """
        Verifies if the kernel entry is trivial (to be set to `1.0`) or not.

        Args:
            i: row index of the entry in the kernel matrix.
            j: column index of the entry in the kernel matrix.
            x_i: a sample from the dataset that corresponds to the row in the kernel matrix.
            y_j: a sample from the dataset that corresponds to the column in the kernel matrix.
            symmetric: whether it is a symmetric case or not.

        Returns:
            `True` if the entry is trivial, `False` otherwise.
        """
        # if we evaluate all combinations, then it is non-trivial
        if self._evaluate_duplicates == "all":
            return False

        # if we are on the diagonal and we don't evaluate it, it is trivial
        if symmetric and i == j and self._evaluate_duplicates == "off_diagonal":
            return True

        # if don't evaluate any duplicates
        if np.array_equal(x_i, y_j) and self._evaluate_duplicates == "none":
            return True

        # otherwise evaluate
        return False

    @property
    def fidelity(self):
        """Returns the fidelity primitive used by this kernel."""
        return self._fidelity

    @property
    def evaluate_duplicates(self):
        """Returns the strategy used by this kernel to evaluate kernel matrix elements if duplicate
        samples are found."""
        return self._evaluate_duplicates
