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
"""Overlap Quantum Kernel"""

from __future__ import annotations
import warnings
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import BaseStateFidelity, ComputeUncompute

from .base_kernel import BaseKernel


class QuantumKernel(BaseKernel):
    r"""
    QuantumKernel

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
        sampler: Sampler | None = None,
        feature_map: QuantumCircuit | None = None,
        fidelity: str | BaseStateFidelity = "zero_prob",
        enforce_psd: bool = True,
        evaluate_duplicates: str = "off_diagonal",
    ) -> None:
        super().__init__(enforce_psd=enforce_psd)

        if feature_map is None:
            feature_map = ZZFeatureMap(2)

        self._num_features = feature_map.num_parameters
        eval_duplicates = evaluate_duplicates.lower()
        if eval_duplicates not in ("all", "off_diagonal", "none"):
            raise ValueError(
                f"Unsupported value passed as evaluate_duplicates: {evaluate_duplicates}"
            )
        self._evaluate_duplicates = eval_duplicates

        if isinstance(fidelity, str):
            if sampler is None:
                raise ValueError(
                    "If the fidelity is passed as a string, a sampler has to be provided"
                    "(currently set to None)."
                )
            if fidelity == "zero_prob":
                self._fidelity = ComputeUncompute(
                    sampler=sampler, left_circuit=feature_map, right_circuit=feature_map
                )
            else:
                raise ValueError(
                    f"{fidelity} is not a valid string for a fidelity. Currently supported: 'zero_prob'."
                )
        else:
            if sampler is not None:
                warnings.warn(
                    "Passed both a sampler and a fidelity instance. If passing a fidelity instance"
                    " for `fidelity`, the sampler will not be used.",
                )
            self._fidelity = fidelity
            fidelity.set_circuits(left_circuit=feature_map, right_circuit=feature_map)

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        x_vec, y_vec = self._check_and_reshape(x_vec, y_vec)
        is_symmetric = np.all(x_vec == y_vec)
        shape = len(x_vec), len(y_vec)

        if self._evaluate_duplicates == "off_diagonal" and is_symmetric:
            # diagonal entries are trivial
            trivial_entries = np.eye(shape[0], dtype=bool)
        elif self._evaluate_duplicates == "none":
            # entries with same parameters are trivial
            trivial_entries = np.array([[np.all(x_i == y_j) for y_j in y_vec] for x_i in x_vec])
        else:
            # no entries are trivial
            trivial_entries = np.zeros(shape, dtype=bool)

        if is_symmetric:
            left_parameters, right_parameters = self._get_symmetric_parametrization(
                x_vec, trivial_entries
            )
            kernel_matrix = self._get_symmetric_kernel_matrix(
                left_parameters, right_parameters, trivial_entries
            )

        else:
            left_parameters, right_parameters = self._get_parametrization(
                x_vec, y_vec, trivial_entries
            )
            kernel_matrix = self._get_kernel_matrix(
                left_parameters, right_parameters, trivial_entries
            )

        if is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)
        return kernel_matrix

    def _get_parametrization(
        self, x_vec: np.ndarray, y_vec: np.ndarray, trivial_entries: np.ndarray
    ) -> tuple[np.ndarray]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        x_count = x_vec.shape[0]
        y_count = y_vec.shape[0]

        num_trivials = np.sum(trivial_entries)

        left_parameters = np.zeros((x_count * y_count - num_trivials, x_vec.shape[1]))
        right_parameters = np.zeros((x_count * y_count - num_trivials, y_vec.shape[1]))
        index = 0
        for i, x_i in enumerate(x_vec):
            for j, y_j in enumerate(y_vec):
                if trivial_entries[i, j]:
                    # trivial entries are not sent to the parameter lists
                    continue
                left_parameters[index, :] = x_i
                right_parameters[index, :] = y_j
                index += 1

        return left_parameters, right_parameters

    def _get_symmetric_parametrization(
        self, x_vec: np.ndarray, trivial_entries: np.ndarray
    ) -> tuple[np.ndarray]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        x_count = x_vec.shape[0]

        # count the trivial entries on the upper triangular matrix
        num_trivials = np.sum(np.triu(trivial_entries))

        left_parameters = np.zeros((x_count * (x_count + 1) // 2 - num_trivials, x_vec.shape[1]))
        right_parameters = np.zeros((x_count * (x_count + 1) // 2 - num_trivials, x_vec.shape[1]))

        index = 0
        for i, x_i in enumerate(x_vec):
            for j, x_j in enumerate(x_vec[i:]):
                if trivial_entries[i, j + i]:
                    # trivial entries are not sent to the parameter lists
                    continue
                left_parameters[index, :] = x_i
                right_parameters[index, :] = x_j
                index += 1

        return left_parameters, right_parameters

    def _get_kernel_matrix(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray, trivial_entries: np.ndarray
    ) -> np.ndarray:
        """
        Given a parametrization, this computes the symmetric kernel matrix.
        """
        if np.all(trivial_entries):
            return trivial_entries
        kernel_entries = self._fidelity(left_parameters, right_parameters)
        kernel_matrix = np.zeros(trivial_entries.shape)

        index = 0
        for i in range(trivial_entries.shape[0]):
            for j in range(trivial_entries.shape[1]):
                if trivial_entries[i, j]:
                    kernel_matrix[i, j] = 1.0
                else:
                    kernel_matrix[i, j] = kernel_entries[index]
                    index += 1
        return kernel_matrix

    def _get_symmetric_kernel_matrix(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray, trivial_entries: np.ndarray
    ) -> np.ndarray:
        """
        Given a set of parametrization, this computes the kernel matrix.
        """
        if np.all(trivial_entries):
            return trivial_entries
        kernel_entries = self._fidelity(left_parameters, right_parameters)
        kernel_matrix = np.zeros(trivial_entries.shape)
        index = 0
        for i in range(trivial_entries.shape[0]):
            for j in range(i, trivial_entries.shape[1]):
                if trivial_entries[i, j]:
                    kernel_matrix[i, j] = 1.0
                else:
                    kernel_matrix[i, j] = kernel_entries[index]
                    index += 1

        kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(kernel_matrix.diagonal())
        return kernel_matrix
