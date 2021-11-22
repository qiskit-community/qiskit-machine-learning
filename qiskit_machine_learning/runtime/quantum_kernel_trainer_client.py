# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Qiskit Machine Learning Quantum Kernel Trainer Runtime Client."""

from typing import Optional, Union, Dict, Callable, Any, Iterable
import copy

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit.algorithms.optimizers import Optimizer, SPSA
from qiskit.utils.algorithm_globals import algorithm_globals

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainerResult


class QKTRuntimeResult(QuantumKernelTrainerResult):
    """The QKTRuntimeClient result object.

    This result objects contains the same as the QuantumKernelTrainerResult and additionally the history
    of the optimizer, containing information such as the function and parameter values per step.
    """

    def __init__(self) -> None:
        super().__init__()
        self._job_id = None  # type: str
        self._optimizer_history = None  # type: Dict[str, Any]

    @property
    def job_id(self) -> str:
        """The job ID associated with the VQE runtime job."""
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        """Set the job ID associated with the VQE runtime job."""
        self._job_id = job_id

    @property
    def optimizer_history(self) -> Optional[Dict[str, Any]]:
        """The optimizer history."""
        return self._optimizer_history

    @optimizer_history.setter
    def optimizer_history(self, history: Dict[str, Any]) -> None:
        """Set the optimizer history."""
        self._optimizer_history = history


class QuantumKernelTrainerClient:
    """The Quantum Kernel Trainer Runtime Client.

    This class is a client to call the quantum-kernel-trainer program in Qiskit Runtime."""

    def __init__(
        self,
        quantum_kernel: QuantumKernel,
        loss: Callable,
        optimizer: Optional[Optimizer] = SPSA(),
        initial_point: Optional[np.ndarray] = None,
        initial_layout: Optional[Iterable[int]] = None,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
        shots: int = 1024,
        measurement_error_mitigation: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
    ) -> None:
        """
        Args:
            quantum_kernel: Quantum kernel to optimize. The quantum kernel feature map
                will have its user parameters unbound before submitting to runtime program
                to avoid conflicts between internal class fields when decoding.
            loss: (str): Loss functions available via string: {'svc_loss: SVCLoss()}
            optimizer: An instance of ``Optimizer`` to be used in training. Defaults to
                ``SPSA``.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` a random vector is used.
            provider: Provider that supports the runtime feature.
            backend: The backend to run the circuits on.
            shots: The number of shots to be used
            measurement_error_mitigation: Whether or not to use measurement error mitigation.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
        """
        # define program name
        self._program_id = "quantum-kernel-training"

        # store settings
        self._quantum_kernel = quantum_kernel
        self._loss = loss
        self._optimizer = optimizer
        self._initial_point = initial_point
        self._initial_layout = initial_layout
        self._provider = None
        self._backend = backend
        self._shots = shots
        self._measurement_error_mitigation = measurement_error_mitigation
        self._callback = callback

        # use setter to check for valid inputs
        if provider is not None:
            self.provider = provider

    @property
    def provider(self) -> Optional[Provider]:
        """Return the provider."""
        return self._provider

    @provider.setter
    def provider(self, provider: Provider) -> None:
        """Set the provider. Must be a provider that supports the runtime feature."""
        try:
            _ = hasattr(provider, "runtime")
        except QiskitError:
            # pylint: disable=raise-missing-from
            raise ValueError(f"The provider {provider} does not provide a runtime environment.")

        self._provider = provider

    @property
    def program_id(self) -> str:
        """Return the program ID."""
        return self._program_id

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Return the quantum kernel."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel) -> None:
        """Set the quantum kernel."""
        self._quantum_kernel = quantum_kernel

    @property
    def loss(self) -> Callable:
        """Return the loss."""
        return self._loss

    @loss.setter
    def loss(self, loss: Callable) -> None:
        """Sets the loss."""
        if callable(loss):
            self._loss = loss
        else:
            raise TypeError(
                "The loss function must be an instance of ``KernelLoss`` or ``Callable``."
            )

    @property
    def optimizer(self) -> Union[Optimizer, Dict[str, Any]]:
        """Return the dictionary describing the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        """Set the optimizer."""
        if not isinstance(optimizer, Optimizer):
            raise TypeError("The optimizer must be an instance of qiskit.algorithms.optimizers")
        self._optimizer = optimizer

    @property
    def backend(self) -> Optional[Backend]:
        """Returns the backend."""
        return self._backend

    @backend.setter
    def backend(self, backend) -> None:
        """Sets the backend."""
        self._backend = backend

    @property
    def shots(self) -> int:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, shots: int) -> None:
        """Set the number of shots."""
        self._shots = shots

    @property
    def measurement_error_mitigation(self) -> bool:
        """Returns whether or not to use measurement error mitigation.
        Readout error mitigation is done using a complete measurement fitter with the
        ``self.shots`` number of shots and re-calibrations every 30 minutes.
        """
        return self._measurement_error_mitigation

    @measurement_error_mitigation.setter
    def measurement_error_mitigation(self, measurement_error_mitigation: bool) -> None:
        """Whether or not to use readout error mitigation."""
        self._measurement_error_mitigation = measurement_error_mitigation

    @property
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """Sets the initial point."""
        self._initial_point = initial_point

    @property
    def initial_layout(self) -> Optional[Iterable[int]]:
        """Returns the initial layout."""
        return self._initial_layout

    @initial_layout.setter
    def initial_layout(self, initial_layout: Optional[Iterable[int]]) -> None:
        """Sets the initial layout."""
        self._initial_layout = initial_layout

    def fit_kernel(
        self,
        data: Iterable[float],
        labels: Iterable[float],
    ) -> QKTRuntimeResult:
        """Calls the Quantum Kernel Training Runtime ('quantum-kernel-training') to train
        the quantum kernel.

        Args:
            data: A 2D array representing an ``(N, M)`` training dataset
                    ``N`` = number of samples
                    ``M`` = feature dimension
            labels: A length-N array of training labels
        Returns:
            QKTRuntimeResult: A :class:`~qiskit_machine_learning.runtime.QKTRuntimeResult`
                                object containing the results of kernel training.

        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            RuntimeError: If the job execution failed.
        """
        num_params = len(self._quantum_kernel.user_parameters)
        if num_params == 0:
            msg = "Quantum kernel cannot be fit because there are no user parameters specified."
            raise ValueError(msg)

        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if self._provider is None:
            raise ValueError("The provider has not been set.")

        # The quantum kernel feature map should be unbound before sending to runtime
        # to avoid conflicts between the feature map, unbound feature map, and the
        # user_param_binds fields when object is decoded.
        unbind_params = list(self._quantum_kernel.user_parameters)
        self._quantum_kernel.assign_user_parameters(unbind_params)

        # Randomly initialize the initial point if one was not passed
        if self._initial_point is None:
            self._initial_point = algorithm_globals.random.random(num_params)

        if self._initial_layout is None:
            self._initial_layout = list(range(self._quantum_kernel.feature_map.num_qubits))

        inputs = {
            "quantum_kernel": self._quantum_kernel,
            "data": data,
            "labels": labels,
            "optimizer": self._optimizer,
            "shots": self._shots,
            "measurement_error_mitigation": self._measurement_error_mitigation,
            "initial_point": self._initial_point,
        }

        # define runtime options
        options = {"backend_name": self._backend.name(), "initial_layout": self._initial_layout}

        # send job to runtime and return result
        job = self._provider.runtime.run(
            program_id=self._program_id,
            inputs=inputs,
            options=options,
        )

        # print job ID if something goes wrong
        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        # re-build result from serialized return value
        qkt_result = QKTRuntimeResult()
        qkt_result.job_id = job.job_id()
        qkt_result.optimal_parameters = result.get("optimal_parameters", None)
        qkt_result.optimal_point = result.get("optimal_point", None)
        qkt_result.optimal_value = result.get("optimal_value", None)
        qkt_result.optimizer_evals = result.get("optimizer_evals", None)
        qkt_result.optimizer_time = result.get("optimizer_time", None)
        qkt_result.optimizer_history = result.get("optimizer_history", None)

        # Return a trained kernel in the results dict
        qkernel_out = copy.copy(self._quantum_kernel)
        qkernel_out.assign_user_parameters(qkt_result.optimal_point)
        qkt_result.quantum_kernel = qkernel_out

        return qkt_result
