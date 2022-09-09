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

"""Trainable Quantum Kernel"""

from __future__ import annotations
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import BaseStateFidelity

from qiskit_machine_learning import QiskitMachineLearningError
from .quantum_kernel import QuantumKernel
from .trainable_kernel_mixin import TrainableKernelMixin


class TrainableQuantumKernel(TrainableKernelMixin, QuantumKernel):
    r"""
    Finding good quantum kernels for a specific machine learning task
    is a big challenge in quantum machine learning. One way to choose
    the kernel is to add trainable parameters to the feature map, which
    can be used to fine-tune the kernel.

    This kernel has trainable parameters :math:`\theta` that can be bound
    using training algorithms. The kernel entries are given as

    .. math::

        K_{\theta}(x,y) = |\langle \phi_{\theta}(x) | \phi_{\theta}(y) \rangle|^2
    """

    def __init__(
        self,
        sampler: Sampler | None = None,
        feature_map: QuantumCircuit | None = None,
        fidelity: str | BaseStateFidelity = "zero_prob",
        training_parameters: ParameterVector | list[Parameter] | None = None,
        enforce_psd: bool = True,
    ) -> None:
        super().__init__(
            sampler,
            feature_map,
            fidelity=fidelity,
            training_parameters=training_parameters,
            enforce_psd=enforce_psd,
        )

        # self._num_features = self._num_features - self.num_parameters
        self._num_features = feature_map.num_parameters - self.num_parameters
        self._feature_parameters = [
            parameter
            for parameter in feature_map.parameters
            if parameter not in training_parameters
        ]
        self._parameter_dict = {parameter: None for parameter in feature_map.parameters}

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray = None) -> np.ndarray:
        for param in self._training_parameters:
            if self._parameter_dict[param] is None:
                raise QiskitMachineLearningError(
                    f"Trainable parameter {param} has not been bound. Make sure to bind all"
                    "trainable parameters to numerical values using `.assign_training_parameters()`"
                    "before calling `.evaluate()`."
                )
        return super().evaluate(x_vec, y_vec)

    def _parameter_array(self, x_vec: np.ndarray) -> np.ndarray:
        """
        Combines the feature values and the trainable parameters into one array.
        """
        full_array = np.zeros((x_vec.shape[0], self._num_features + self.num_parameters))
        for i, x in enumerate(x_vec):
            self._parameter_dict.update(
                {feature_param: x[j] for j, feature_param in enumerate(self._feature_parameters)}
            )
            full_array[i, :] = list(self._parameter_dict.values())
        return full_array

    def _get_parametrization(
        self, x_vec: np.ndarray, y_vec: np.ndarray, trivial_entries: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        new_x_vec = self._parameter_array(x_vec)
        new_y_vec = self._parameter_array(y_vec)

        return super()._get_parametrization(new_x_vec, new_y_vec, trivial_entries)

    def _get_symmetric_parametrization(
        self, x_vec: np.ndarray, trivial_entries: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        new_x_vec = self._parameter_array(x_vec)

        return super()._get_symmetric_parametrization(new_x_vec, trivial_entries)
