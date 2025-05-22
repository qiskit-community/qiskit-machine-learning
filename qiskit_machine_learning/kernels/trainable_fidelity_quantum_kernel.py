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

"""Trainable Quantum Kernel"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from ..state_fidelities import BaseStateFidelity

from .fidelity_quantum_kernel import FidelityQuantumKernel, KernelIndices
from .trainable_kernel import TrainableKernel


class TrainableFidelityQuantumKernel(TrainableKernel, FidelityQuantumKernel):
    r"""
    An implementation of the quantum kernel that is based on the
    :class:`~qiskit_machine_learning.state_fidelities.BaseStateFidelity` algorithm
    and provides ability to train it.

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
        fidelity: BaseStateFidelity | None = None,
        training_parameters: ParameterVector | Sequence[Parameter] | None = None,
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
                :class:`~qiskit_machine_learning.state_fidelities.BaseStateFidelity` primitive to be used
                to compute fidelity between states. Default is
                :class:`~qiskit_machine_learning.state_fidelities.ComputeUncompute` which is created on
                top of the reference sampler defined by :class:`~qiskit.primitives.Sampler`.
            training_parameters: Iterable containing :class:`~qiskit.circuit.Parameter` objects
                which correspond to quantum gates on the feature map circuit which may be tuned.
                If users intend to tune feature map parameters to find optimal values, this field
                should be set.
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
        """
        super().__init__(
            feature_map=feature_map,
            fidelity=fidelity,
            training_parameters=training_parameters,
            enforce_psd=enforce_psd,
            evaluate_duplicates=evaluate_duplicates,
        )

        # override the num of features defined in the base class
        self._num_features = self.feature_map.num_parameters - self._num_training_parameters
        self._feature_parameters = [
            parameter
            for parameter in self.feature_map.parameters
            if parameter not in self._training_parameters
        ]
        self._parameter_dict = {parameter: None for parameter in self.feature_map.parameters}

    def _get_parameterization(
        self, x_vec: np.ndarray, y_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, KernelIndices]:
        new_x_vec = self._parameter_array(x_vec)
        new_y_vec = self._parameter_array(y_vec)

        return super()._get_parameterization(new_x_vec, new_y_vec)

    def _get_symmetric_parameterization(
        self, x_vec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, KernelIndices]:
        new_x_vec = self._parameter_array(x_vec)

        return super()._get_symmetric_parameterization(new_x_vec)
