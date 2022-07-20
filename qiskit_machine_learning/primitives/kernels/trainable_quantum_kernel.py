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
from qiskit.primitives.fidelity import BaseFidelity

from qiskit_machine_learning.utils import make_2d
from .quantum_kernel import QuantumKernel


class TrainableQuantumKernel(QuantumKernel):
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
        fidelity: str | BaseFidelity = "zero_prob",
        training_parameters: ParameterVector | list[Parameter] | None = None,
        enforce_psd: bool = True,
    ) -> None:
        super().__init__(sampler, feature_map, fidelity=fidelity, enforce_psd=enforce_psd)
        if training_parameters is None:
            self._training_parameters = []

        self.num_parameters = len(training_parameters)
        self._num_features = self._num_features - self.num_parameters
        self._training_parameters = training_parameters
        self._feature_parameters = feature_map.parameters - training_parameters
        self._parameter_dict = {parameter: None for parameter in feature_map.parameters}

        self.parameter_values = np.zeros(self.num_parameters)

    def assign_training_parameters(self, parameter_values: np.ndarray) -> None:
        """
        Fix the training parameters to numerical values.
        """
        if not isinstance(parameter_values, dict):
            if len(parameter_values) != self.num_parameters:
                raise ValueError(
                    f"The number of given parameters is wrong ({len(parameter_values)}),"
                    f"expected {self.num_parameters}."
                )
            self._parameter_dict.update(
                {
                    parameter: parameter_values[i]
                    for i, parameter in enumerate(self._training_parameters)
                }
            )
        else:
            for key in parameter_values:
                if key not in self._training_parameters:
                    raise ValueError(
                        f"Parameter {key} is not a trainable parameter of the feature map and"
                        f"thus cannot be bound. Make sure {key} is provided in the the trainable"
                        "parameters when initializing the kernel."
                    )
                self._parameter_dict[key] = parameter_values[key]

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

    def _get_parametrization(self, x_vec: np.ndarray, y_vec: np.ndarray) -> tuple[np.ndarray]:
        new_x_vec = self._parameter_array(x_vec)
        new_y_vec = self._parameter_array(y_vec)

        return super()._get_parametrization(new_x_vec, new_y_vec)

    def _get_symmetric_parametrization(self, x_vec: np.ndarray) -> np.ndarray:
        new_x_vec = self._parameter_array(x_vec)

        return super()._get_symmetric_parametrization(new_x_vec)
