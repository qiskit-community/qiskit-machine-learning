# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
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

from collections.abc import Sequence
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from ..state_fidelities import BaseStateFidelity, ComputeUncompute

from .base_kernel import BaseKernel

KernelIndices = List[Tuple[int, int]]


class FidelityQuantumKernel(BaseKernel):
    r"""
    An implementation of the quantum kernel interface based on the
    :class:`~qiskit_machine_learning.state_fidelities.BaseStateFidelity` algorithm.

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
        max_circuits_per_job: int = None,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
            fidelity: An instance of the
                :class:`~qiskit_machine_learning.state_fidelities.BaseStateFidelity` primitive to be used
                to compute fidelity between states. Default is
                :class:`~qiskit_machine_learning.state_fidelities.ComputeUncompute` which is created on
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
            max_circuits_per_job: Maximum number of circuits per job for the backend. Please
               check the backend specifications. Use ``None`` for all entries per job. Default ``None``.
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
        if max_circuits_per_job is not None:
            if max_circuits_per_job < 1:
                raise ValueError(
                    f"Unsupported value passed as max_circuits_per_job: {max_circuits_per_job}"
                )
        self.max_circuits_per_job = max_circuits_per_job

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

        indices = np.asarray(
            [
                (i, j)
                for i, x_i in enumerate(x_vec)
                for j, y_j in enumerate(y_vec)
                if not self._is_trivial(i, j, x_i, y_j, False)
            ]
        )

        if indices.size > 0:
            left_parameters = x_vec[indices[:, 0]]
            right_parameters = y_vec[indices[:, 1]]

        return left_parameters, right_parameters, indices.tolist()

    def _get_symmetric_parameterization(
        self, x_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, KernelIndices]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        num_features = x_vec.shape[1]
        left_parameters = np.zeros((0, num_features))
        right_parameters = np.zeros((0, num_features))

        indices = np.asarray(
            [
                (i, i + j)
                for i, x_i in enumerate(x_vec)
                for j, x_j in enumerate(x_vec[i:])
                if not self._is_trivial(i, i + j, x_i, x_j, True)
            ]
        )

        if indices.size > 0:
            left_parameters = x_vec[indices[:, 0]]
            right_parameters = x_vec[indices[:, 1]]

        return left_parameters, right_parameters, indices.tolist()

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

    def _get_kernel_entries(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray
    ) -> Sequence[float]:
        """
        Gets kernel entries by executing the underlying fidelity instance and getting the results
        back from the async job.
        """
        num_circuits = left_parameters.shape[0]
        kernel_entries = []
        # Check if it is trivial case, only identical samples
        if num_circuits != 0:
            if self.max_circuits_per_job is None:
                job = self._fidelity.run(
                    [self._feature_map] * num_circuits,
                    [self._feature_map] * num_circuits,
                    left_parameters,  # type: ignore[arg-type]
                    right_parameters,  # type: ignore[arg-type]
                )
                kernel_entries = job.result().fidelities
            else:
                # Determine the number of chunks needed
                num_chunks = (
                    num_circuits + self.max_circuits_per_job - 1
                ) // self.max_circuits_per_job
                for i in range(num_chunks):
                    # Determine the range of indices for this chunk
                    start_idx = i * self.max_circuits_per_job
                    end_idx = min((i + 1) * self.max_circuits_per_job, num_circuits)
                    # Extract the parameters for this chunk
                    chunk_left_parameters = left_parameters[start_idx:end_idx]
                    chunk_right_parameters = right_parameters[start_idx:end_idx]
                    # Execute this chunk
                    job = self._fidelity.run(
                        [self._feature_map] * (end_idx - start_idx),
                        [self._feature_map] * (end_idx - start_idx),
                        chunk_left_parameters,  # type: ignore[arg-type]
                        chunk_right_parameters,  # type: ignore[arg-type]
                    )
                    # Extend the kernel_entries list with the results from this chunk
                    kernel_entries.extend(job.result().fidelities)
        return kernel_entries

    # pylint: disable=too-many-positional-arguments
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
