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

from typing import Tuple, Optional, Callable, List, Union
import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.primitives.fidelity import BaseFidelity


from qiskit_machine_learning.primitives.kernels import QuantumKernel
from qiskit_machine_learning.utils import make_2d

SamplerFactory = Callable[[List[QuantumCircuit]], Sampler]


class TrainableQuantumKernel(QuantumKernel):
    """
    Trainable overlap kernel.
    """

    def __init__(
        self,
        sampler_factory: SamplerFactory,
        feature_map: Optional[QuantumCircuit] = None,
        *,
        fidelity: Union[str, BaseFidelity] = "zero_prob",
        num_training_parameters: int = 0,
        enforce_psd: bool = True,
    ) -> None:
        super().__init__(sampler_factory, feature_map, fidelity=fidelity, enforce_psd=enforce_psd)
        self.num_parameters = num_training_parameters
        self._num_features = self._num_features - self.num_parameters

        self.parameter_values = np.zeros(self.num_parameters)

    def bind_parameters_values(self, parameter_values: np.ndarray) -> None:
        """
        Fix the training parameters to numerical values.
        """
        if parameter_values.shape == self.parameter_values.shape:
            self.parameter_values = parameter_values
        else:
            raise ValueError(
                f"The given parameters are in the wrong shape {parameter_values.shape}, expected {self.parameter_values.shape}."
            )

    def _get_parametrization(self, x_vec: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray]:
        new_x_vec = np.hstack((x_vec, make_2d(self.parameter_values, len(x_vec))))
        new_y_vec = np.hstack((y_vec, make_2d(self.parameter_values, len(y_vec))))

        return super()._get_parametrization(new_x_vec, new_y_vec)

    def _get_symmetric_parametrization(self, x_vec: np.ndarray) -> np.ndarray:
        new_x_vec = np.hstack((x_vec, make_2d(self.parameter_values, len(x_vec))))

        return super()._get_symmetric_parametrization(new_x_vec)