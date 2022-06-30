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

from typing import Optional, Tuple, Callable, List, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.primitives.fidelity import BaseFidelity, Fidelity

from qiskit_machine_learning.primitives.kernels import BaseKernel


SamplerFactory = Callable[[List[QuantumCircuit]], Sampler]


class QuantumKernel(BaseKernel):
    """
    Overlap Kernel
    """

    def __init__(
        self,
        sampler_factory: SamplerFactory,
        feature_map: Optional[QuantumCircuit] = None,
        *,
        fidelity: Union[str, BaseFidelity] = "zero_prob",
        enforce_psd: bool = True,
    ) -> None:
        super().__init__(sampler_factory, enforce_psd=enforce_psd)

        if feature_map is None:
            feature_map = ZZFeatureMap(2).decompose()

        self._num_features = feature_map.num_parameters

        if isinstance(fidelity, str) and fidelity == "zero_prob":
            self._fidelity = Fidelity()
        else:
            self._fidelity = fidelity
        self._fidelity.set_circuits(feature_map, feature_map)

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        x_vec, y_vec = self._check_and_reshape(x_vec, y_vec)
        is_symmetric = np.all(x_vec == y_vec)
        shape = len(x_vec), len(y_vec)

        if is_symmetric:
            left_parameters, right_parameters = self._get_symmetric_parametrization(x_vec)
            kernel_matrix = self._get_symmetric_kernel_matrix(
                left_parameters, right_parameters, shape
            )

        else:
            left_parameters, right_parameters = self._get_parametrization(x_vec, y_vec)
            kernel_matrix = self._get_kernel_matrix(left_parameters, right_parameters, shape)

        if is_symmetric and self._enforce_psd:
            kernel_matrix = self._make_psd(kernel_matrix)
        return kernel_matrix

    def _get_parametrization(self, x_vec: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray]:
        """
        Combines x_vec and y_vec to get all the combinations needed to evaluate the kernel entries.
        """
        x_count = x_vec.shape[0]
        y_count = y_vec.shape[0]

        left_parameters = np.zeros((x_count * y_count, x_vec.shape[1]))
        right_parameters = np.zeros((x_count * y_count, y_vec.shape[1]))
        index = 0
        for x_i in x_vec:
            for y_j in y_vec:
                left_parameters[index, :] = x_i
                right_parameters[index, :] = y_j
                index += 1

        return left_parameters, right_parameters

    def _get_symmetric_parametrization(self, x_vec: np.ndarray) -> Tuple[np.ndarray]:
        """
        Combines two copies of x_vec to get all the combinations needed to evaluate the kernel entries.
        """
        x_count = x_vec.shape[0]

        left_parameters = np.zeros((x_count * (x_count + 1) // 2, x_vec.shape[1]))
        right_parameters = np.zeros((x_count * (x_count + 1) // 2, x_vec.shape[1]))

        index = 0
        for i, x_i in enumerate(x_vec):
            for x_j in x_vec[i:]:
                left_parameters[index, :] = x_i
                right_parameters[index, :] = x_j
                index += 1

        return left_parameters, right_parameters

    def _get_kernel_matrix(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray, shape: Tuple
    ) -> np.ndarray:
        """
        Given a parametrization, this computes the symmetric kernel matrix.
        """
        kernel_entries = self._fidelity.compute(left_parameters, right_parameters)
        kernel_matrix = np.zeros(shape)

        index = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                kernel_matrix[i, j] = kernel_entries[index]
                index += 1
        return kernel_matrix

    def _get_symmetric_kernel_matrix(
        self, left_parameters: np.ndarray, right_parameters: np.ndarray, shape: Tuple
    ) -> np.ndarray:
        """
        Given a set of parametrization, this computes the kernel matrix.
        """
        kernel_entries = self._fidelity.compute(left_parameters, right_parameters)
        kernel_matrix = np.zeros(shape)
        index = 0
        for i in range(shape[0]):
            for j in range(i, shape[1]):
                kernel_matrix[i, j] = kernel_entries[index]
                index += 1

        kernel_matrix = kernel_matrix + kernel_matrix.T - np.diag(kernel_matrix.diagonal())
        return kernel_matrix

    def __enter__(self):
        """
        Creates the full fidelity class by creating the sampler from the factory.
        """
        self._fidelity.sampler_from_factory(self._sampler_factory)
        return self

    def __exit__(self, *args):
        """
        Closes the sampler session.
        """
        self._fidelity.sampler.close()
