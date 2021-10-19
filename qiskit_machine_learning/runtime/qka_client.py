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

"""The Qiskit Machine Learning QKA Runtime Client."""

from typing import Optional, Union, Dict, Callable, Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit.algorithms.variational_algorithm import VariationalResult
from qiskit.algorithms.optimizers import Optimizer, SPSA


class QKAClient:
    """The Qiskit Machine Learning Runtime Client.

    This class is a client to call the QKA program in Qiskit Runtime."""

    def __init__(
        self,
        feature_map: QuantumCircuit,
        optimizer: Optional[Union[Optimizer, Dict[str, Any]]] = None,
        initial_point: Optional[np.ndarray] = None,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
    ) -> None:
        """
        Args:
            optimizer: An optimizer or dictionary specifying a classical optimizer.
                If a dictionary, only SPSA is supported. The dictionary must contain a
                key ``name`` for the name of the optimizer and may contain additional keys for the
                settings. E.g. ``{'name': 'SPSA', 'maxiter': 100}``.
                Per default, SPSA is used.
            backend: The backend to run the circuits on.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` a random vector is used.
            provider: Provider that supports the runtime feature.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                ansatz, the evaluated mean and the evaluated standard deviation.
        """
        if optimizer is None:
            optimizer = SPSA()

        # define program name
        self._program_id = "qka"

        # store settings
        self._provider = None
        self._feature_map = feature_map
        self._optimizer = None
        self._backend = backend
        self._initial_point = initial_point
        self._callback = callback

        # use setter to check for valid inputs
        if provider is not None:
            self.provider = provider

        self.optimizer = optimizer

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
    def feature_map(self) -> QuantumCircuit:
        """Return the ansatz."""
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: QuantumCircuit) -> None:
        """Set the feature map."""
        self._feature_map = feature_map

    @property
    def optimizer(self) -> Union[Optimizer, Dict[str, Any]]:
        """Return the dictionary describing the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Union[Optimizer, Dict[str, Any]]) -> None:
        """Set the optimizer."""
        if isinstance(optimizer, Optimizer):
            self._optimizer = optimizer
        else:
            if "name" not in optimizer.keys():
                raise ValueError(
                    "The optimizer dictionary must contain a ``name`` key specifying the type "
                    "of the optimizer."
                )

            _validate_optimizer_settings(optimizer)

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
    def initial_point(self) -> Optional[np.ndarray]:
        """Returns the initial point."""
        return self._initial_point

    @initial_point.setter
    def initial_point(self, initial_point: Optional[np.ndarray]) -> None:
        """Sets the initial point."""
        self._initial_point = initial_point

    @property
    def callback(self) -> Callable:
        """Returns the callback."""
        return self._callback

    @callback.setter
    def callback(self, callback: Callable) -> None:
        """Set the callback."""
        self._callback = callback

    def align_kernel(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        maxiters: int = 1,
        C: float = 1.0,
        initial_layout: Optional[Iterable[int]] = None,
    ) -> QKARuntimeResult:
        """ """
        if self.backend is None:
            raise ValueError("The backend has not been set.")

        if self.provider is None:
            raise ValueError("The provider has not been set.")

        # combine the settings with the given operator to runtime inputs
        inputs = {
            "feature_map": self.feature_map,
            "data": data,
            "labels": labels,
            "optimizer": self.optimizer,
            "initial_point": self.initial_point,
        }

        # define runtime options
        options = {"backend_name": self.backend.name()}

        # send job to runtime and return result
        cb_qka = QKACallback()
        job = self.provider.runtime.run(
            program_id=self.program_id,
            inputs=inputs,
            options=options,
            callback=cb_qka.callback,
        )
        # print job ID if something goes wrong
        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        # re-build result from serialized return value
        qka_result = QKARuntimeResult()
        qka_result.job_id = job.job_id()
        qka_result.optimal_parameters = result.get("optimal_parameters", None)
        qka_result.optimal_point = result.get("optimal_point", None)
        qka_result.optimal_value = result.get("optimal_value", None)
        qka_result.optimizer_evals = result.get("optimizer_evals", None)
        qka_result.optimizer_time = result.get("optimizer_time", None)
        qka_result.optimizer_history = result.get("optimizer_history", None)

        return qka_result


class QKARuntimeResult(VariationalResult):
    """The QKAClient result object.

    This result objects contains the same as the QKAResult and additionally the history
    of the optimizer, containing information such as the function and parameter values per step.
    """

    def __init__(self) -> None:
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


def _validate_optimizer_settings(settings):
    name = settings.get("name", None)
    if name not in ["SPSA"]:
        raise NotImplementedError("Only SPSA is currently supported.")

    allowed_settings = [
        "name",
        "maxiter",
        "blocking",
        "allowed_increase",
        "trust_region",
        "learning_rate",
        "perturbation",
        "resamplings",
        "last_avg",
        "second_order",
        "hessian_delay",
        "regularization",
        "initial_hessian",
    ]

    unsupported_args = set(settings.keys()) - set(allowed_settings)

    if len(unsupported_args) > 0:
        raise ValueError(
            f"The following settings are unsupported for the {name} optimizer: "
            f"{unsupported_args}"
        )


class QKACallback:
    """Callback wrapper class."""

    def __init__(self) -> None:
        self._data = [[] for i in range(5)]

    def callback(self, x0, x1=None, x2=None, x3=None, x4=None):
        """
        x[0]: number of function evaluations
        x[1]: the parameters
        x[2]: the function value
        x[3]: the stepsize
        x[4]: whether the step was accepted
        """
        self._data[0].append(x0)
        self._data[1].append(x1)
        self._data[2].append(x2)
        self._data[3].append(x3)
        self._data[4].append(x4)

    def get_callback_data(self):
        return self._data

    def clear_callback_data(self):
        self._data = [[] for i in range(5)]
