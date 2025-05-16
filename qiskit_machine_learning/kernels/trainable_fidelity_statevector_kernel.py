# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2023, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Trainable Fidelity Statevector Kernel"""

from __future__ import annotations

from typing import Sequence, Type

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector


from .fidelity_statevector_kernel import FidelityStatevectorKernel, SV
from .trainable_kernel import TrainableKernel


class TrainableFidelityStatevectorKernel(TrainableKernel, FidelityStatevectorKernel):
    r"""
    A trainable version of the
    :class:`~qiskit_machine_learning.kernels.FidelityStatevectorKernel`.

    Finding good quantum kernels for a specific machine learning task is a big challenge in quantum
    machine learning. One way to choose the kernel is to add trainable parameters to the feature
    map, which can be used to fine-tune the kernel.

    This kernel has trainable parameters :math:`\theta` that can be bound using training algorithms.
    The kernel entries are given as

    .. math::

        K_{\theta}(x,y) = |\langle \phi_{\theta}(x) | \phi_{\theta}(y) \rangle|^2
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        statevector_type: Type[SV] = Statevector,
        training_parameters: ParameterVector | Sequence[Parameter] | None = None,
        cache_size: int | None = None,
        auto_clear_cache: bool = True,
        shots: int | None = None,
        enforce_psd: bool = True,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. If ``None`` is given,
                :class:`~qiskit.circuit.library.ZZFeatureMap` is used with two qubits. If there's
                a mismatch in the number of qubits of the feature map and the number of features
                in the dataset, then the kernel will try to adjust the feature map to reflect the
                number of features.
            statevector_type: The type of Statevector that will be instantiated using the
                ``feature_map`` quantum circuit and used to compute the fidelity kernel. This type
                should inherit from (and defaults to) :class:`~qiskit.quantum_info.Statevector`.
            training_parameters: Iterable containing :class:`~qiskit.circuit.Parameter` objects
                which correspond to quantum gates on the feature map circuit which may be tuned.
                If users intend to tune feature map parameters to find optimal values, this field
                should be set.
            cache_size: Maximum size of the statevector cache. When ``None`` this is unbounded.
            auto_clear_cache: Determines whether the statevector cache is retained when
                :meth:`evaluate` is called. The cache is automatically cleared by default.
            shots: The number of shots. If ``None``, the exact fidelity is used. Otherwise, the
                mean is taken of samples drawn from a binomial distribution with probability equal
                to the exact fidelity.
            enforce_psd: Project to the closest positive semidefinite matrix if ``x = y``.
                Default ``True``.
        """
        super().__init__(
            feature_map=feature_map,
            statevector_type=statevector_type,
            training_parameters=training_parameters,
            cache_size=cache_size,
            auto_clear_cache=auto_clear_cache,
            shots=shots,
            enforce_psd=enforce_psd,
        )

        # Override the number of features defined in the base class.
        self._num_features = self.feature_map.num_parameters - self._num_training_parameters
        self._feature_parameters = [
            parameter
            for parameter in self.feature_map.parameters
            if parameter not in self._training_parameters
        ]
        self._parameter_dict = {parameter: None for parameter in self.feature_map.parameters}

    def _evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray, is_symmetric: bool):
        new_x_vec = self._parameter_array(x_vec)
        new_y_vec = self._parameter_array(y_vec)
        return super()._evaluate(new_x_vec, new_y_vec, is_symmetric)
