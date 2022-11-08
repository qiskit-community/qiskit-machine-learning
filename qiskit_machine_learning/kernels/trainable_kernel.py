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

from abc import ABC
from typing import Mapping, Sequence

import numpy as np
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parameterexpression import ParameterValueType

from .base_kernel import BaseKernel


class TrainableKernel(BaseKernel, ABC):
    """An abstract definition of the ability to train kernel via specifying training parameters."""

    def __init__(
        self, *, training_parameters: ParameterVector | Sequence[Parameter] | None = None, **kwargs
    ) -> None:
        """
        Args:
            training_parameters: a sequence of training parameters.
            **kwargs: Additional parameters may be used by the super class.
        """
        super().__init__(**kwargs)

        if training_parameters is None:
            training_parameters = []

        self._training_parameters = training_parameters
        self._num_training_parameters = len(self._training_parameters)

        self._parameter_dict = {parameter: None for parameter in training_parameters}

    def assign_training_parameters(
        self,
        parameter_values: Mapping[Parameter, ParameterValueType] | Sequence[ParameterValueType],
    ) -> None:
        """
        Fix the training parameters to numerical values.
        """
        if not isinstance(parameter_values, dict):
            if len(parameter_values) != self._num_training_parameters:
                raise ValueError(
                    f"The number of given parameters is wrong: {len(parameter_values)}, "
                    f"expected {self._num_training_parameters}."
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
                        f"Parameter {key} is not a trainable parameter of the feature map and "
                        f"thus cannot be bound. Make sure {key} is provided in the the trainable "
                        "parameters when initializing the kernel."
                    )
                self._parameter_dict[key] = parameter_values[key]

    @property
    def parameter_values(self) -> np.ndarray:
        """
        Returns numerical values assigned to the training parameters as a numpy array.
        """
        return np.asarray([self._parameter_dict[param] for param in self._training_parameters])

    @property
    def training_parameters(self) -> ParameterVector | Sequence[Parameter]:
        """
        Returns the vector of training parameters.
        """
        return self._training_parameters

    @property
    def num_training_parameters(self) -> int:
        """
        Returns the number of training parameters.
        """
        return len(self._training_parameters)
