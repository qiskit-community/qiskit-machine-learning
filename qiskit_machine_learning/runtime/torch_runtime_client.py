# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Qiskit Machine Learning Torch Runtime Client"""

import base64
from typing import Any, Callable, Dict, Optional, Union, List
import warnings
import dill

from qiskit.exceptions import QiskitError
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit_machine_learning.deprecation import warn_deprecated, deprecate_function, DeprecatedType
from qiskit_machine_learning.runtime.hookbase import HookBase
import qiskit_machine_learning.optionals as _optionals

if _optionals.HAS_TORCH:
    from torch import Tensor
    from torch.nn import Module, MSELoss
    from torch.nn.modules.loss import _Loss
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
else:

    class DataLoader:  # type: ignore
        """Empty DataLoader class
        Replacement if torch.utils.data.DataLoader is not present.
        """

        pass

    class Tensor:  # type: ignore
        """Empty Tensor class
        Replacement if torch.Tensor is not present.
        """

        pass

    class _Loss:  # type: ignore
        """Empty _Loss class
        Replacement if torch.nn.modules.loss._Loss is not present.
        """

        pass

    class Optimizer:  # type: ignore
        """Empty Optimizer
        Replacement if torch.optim.Optimizer is not present.
        """

        pass

    class Module:  # type: ignore
        """Empty Module class
        Replacement if torch.nn.Module is not present.
        Always fails to initialize
        """

        pass


class TorchRuntimeResult:
    """Deprecation: TorchRuntimeClient result object.

    The result objects contains the state dictionary of the trained model,
    and the training history such as the value of loss function in each epoch.
    """

    def __init__(self) -> None:
        warn_deprecated(
            "0.5.0",
            old_type=DeprecatedType.CLASS,
            old_name="TorchRuntimeResult",
            additional_msg=". You should use QiskitRuntimeService to leverage primitives and runtimes",
            stack_level=3,
        )
        self._job_id: Optional[str] = None
        self._train_history: Optional[List[Dict[str, float]]] = None
        self._val_history: Optional[List[Dict[str, float]]] = None
        self._model_state_dict: Optional[Dict[str, Tensor]] = None
        self._execution_time: Optional[float] = None
        self._score: Optional[float] = None
        self._prediction: Optional[Tensor] = None

    @property
    def job_id(self) -> str:
        """The job ID associated with the Torch runtime job."""
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        """Set the job ID associated with the Torch runtime job."""
        self._job_id = job_id

    @property
    def train_history(self) -> List[Dict[str, float]]:
        """The train history"""
        return self._train_history

    @train_history.setter
    def train_history(self, history: List[Dict[str, float]]) -> None:
        """Set the train history"""
        self._train_history = history

    @property
    def val_history(self) -> List[Dict[str, float]]:
        """Returns the validation history"""
        return self._val_history

    @val_history.setter
    def val_history(self, history: List[Dict[str, float]]) -> None:
        """Sets the validation history"""
        self._val_history = history

    @property
    def model_state_dict(self) -> Dict[str, Tensor]:
        """Returns the state dictionary of trained model"""
        return self._model_state_dict

    @model_state_dict.setter
    def model_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Sets the state dictionary of trained model"""
        self._model_state_dict = state_dict

    @property
    def execution_time(self) -> float:
        """Returns the execution time"""
        return self._execution_time

    @execution_time.setter
    def execution_time(self, time: float) -> None:
        """Sets the execution time"""
        self._execution_time = time

    @property
    def score(self) -> float:
        """Returns the score"""
        return self._score

    @score.setter
    def score(self, score: float) -> None:
        """Sets the score"""
        self._score = score

    @property
    def prediction(self) -> Tensor:
        """Returns the prediction"""
        return self._prediction

    @prediction.setter
    def prediction(self, prediction: Tensor) -> None:
        """Sets the prediction"""
        self._prediction = prediction


@_optionals.HAS_TORCH.require_in_instance
class TorchRuntimeClient:
    """Deprecation: TorchRuntimeClient

    The Qiskit Machine Learning Torch runtime client to call the Torch runtime."""

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_func: Callable,
        epochs: int = 10,
        shots: int = 1024,
        measurement_error_mitigation: bool = False,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        """
        Args:
            model: A PyTorch module to be trained.
            optimizer: A PyTorch optimizer for the model parameters.
            loss_func: A PyTorch-compatible loss function. Can be one of the
                official PyTorch loss functions from ``torch.nn.loss`` or a custom
                function defined by the user.
            epochs: The maximum number of training epochs. By default, 10.
            shots: The number of shots for the quantum backend. By default, 1024.
            measurement_error_mitigation: Whether or not to use measurement error mitigation.
            By default, ``False``.
            provider: IBMQ provider that supports runtime services.
            backend: Selected quantum backend.
        """
        warn_deprecated(
            "0.5.0",
            old_type=DeprecatedType.CLASS,
            old_name="TorchRuntimeClient",
            additional_msg=". You should use QiskitRuntimeService to leverage primitives and runtimes",
            stack_level=3,
        )
        # Store settings
        self._provider = None
        self._model = None
        self._loss_func = None
        self._optimizer = None
        self._epochs = epochs
        self._shots = shots
        self._measurement_error_mitigation = measurement_error_mitigation
        self._backend = backend
        self._score_func = None

        # Use setter to check for valid inputs
        if provider is not None:
            self.provider = provider
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.model = model

    @property
    def provider(self) -> Optional[Provider]:
        """Return the provider."""
        return self._provider

    @provider.setter
    def provider(self, provider: Provider) -> None:
        """Set the provider. Must be a provider that supports the runtime feature.

        Raises:
            ValueError: If the provider does not provide a runtime environment.
        """
        try:
            _ = hasattr(provider, "runtime")
        except QiskitError as exc:
            raise ValueError(
                f"The provider {provider} does not provide a runtime environment."
            ) from exc

        self._provider = provider

    @property
    def model(self) -> Module:
        """Return the model."""
        return self._model

    @model.setter
    def model(self, t_model: Module) -> None:
        """Set the model.

        Raises:
            ValueError: If the model is not an instance of ``torch.nn.Module``.
        """
        if not isinstance(t_model, Module):
            raise ValueError("The model must be an instance of torch.nn.Module")
        self._model = t_model

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optim: Optimizer) -> None:
        """Set the optimizer.

        Raises:
            ValueError: If the optimizer is not an instance of ``torch.optim.Optimizer``.
        """
        if not isinstance(optim, Optimizer):
            raise ValueError("The optimizer must be an instance of torch.optim.Optimizer")
        self._optimizer = optim

    @property
    def loss_func(self) -> Union[_Loss, Callable]:
        """Return the loss function."""
        return self._loss_func

    @loss_func.setter
    def loss_func(self, loss: Union[_Loss, Callable]) -> None:
        """Set the loss function.

        Raises:
            ValueError: If the loss function is not callable.
        """
        if callable(loss):
            self._loss_func = loss
        else:
            raise ValueError(
                "The loss function must be an instance of torch.nn.Loss._Loss or Callable"
            )

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

        measurement error mitigation is done using a complete measurement fitter with the
        ``self.shots`` number of shots and re-calibrations every 30 minutes.
        """
        return self._measurement_error_mitigation

    @measurement_error_mitigation.setter
    def measurement_error_mitigation(self, measurement_error_mitigation: bool) -> None:
        """Whether or not to use measurement error mitigation."""
        self._measurement_error_mitigation = measurement_error_mitigation

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        hooks: Optional[Union[List[HookBase], HookBase]] = None,
        start_epoch: int = 0,
        seed: Optional[int] = None,
    ) -> TorchRuntimeResult:
        """Train the model using the Torch Train Runtime ('torch-train').
        All required data is serialized and sent to the server side.
        After training, model's parameters are updated and an instance of ``TorchRuntimeResult``
        is returned as a result.

        Args:
            train_loader: A PyTorch data loader object containing the training dataset.
            val_loader: A PyTorch data loader object containing the validation dataset.
                If no validation loader is provided, the validation step will be skipped
                during training.
            hooks: a hook class of a List of hook classes to interact with the training loop.
            start_epoch: initial epoch for warm-start training. Default value is ``0``.
            seed: Set the random seed for `torch.manual_seed(seed)`.
        Returns:
            result: A :class:`~qiskit_machine_learning.runtime.TorchRuntimeResult` object
                with the trained model's state dictionary and training history data.
        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            ValueError: If the train_loader is not an instance of DataLoader in PyTorch.
            ValueError: If the val_loader is not an instance of DataLoader in PyTorch.
            ValueError: If one of hooks is not a HookBase type
            RuntimeError: If the job execution failed.
        """
        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if self._provider is None:
            raise ValueError("The provider has not been set.")

        if not isinstance(train_loader, DataLoader):
            raise ValueError("`train_loader` must be an instance of `torch.utils.data.DataLoader`.")

        if val_loader:
            if not isinstance(val_loader, DataLoader):
                raise ValueError(
                    "`val_loader` must be an instance of `torch.utils.data.DataLoader`."
                )

        # serialize using dill
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            serial_model = obj_to_str(self.model)
            serial_optim = obj_to_str(self.optimizer)
            serial_loss = obj_to_str(self.loss_func)
            serial_train_data = obj_to_str(train_loader)
        serial_val_data: Optional[str] = None
        # check hooks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if val_loader:
                serial_val_data = obj_to_str(val_loader)
            if hooks is None:
                serial_hooks = obj_to_str([])
            elif isinstance(hooks, HookBase):
                serial_hooks = obj_to_str([hooks])
            elif isinstance(hooks, list) and all(isinstance(hook, HookBase) for hook in hooks):
                serial_hooks = obj_to_str(hooks)
            else:
                raise ValueError("`hooks` must all be of the HookBase type")
        # combine the settings with the serialized buffers to runtime inputs
        inputs = {
            "model": serial_model,
            "optimizer": serial_optim,
            "loss_func": serial_loss,
            "train_data": serial_train_data,
            "val_data": serial_val_data,
            "shots": self._shots,
            "measurement_error_mitigation": self.measurement_error_mitigation,
            "epochs": self._epochs,
            "start_epoch": start_epoch,
            "hooks": serial_hooks,
            "seed": seed,
        }

        # define runtime options
        options = {"backend_name": self._backend.name()}

        # send job to runtime and return result
        job = self.provider.runtime.run(
            program_id="torch-train",
            inputs=inputs,
            options=options,
        )

        # Raise a runtime error with the job id if something goes wrong
        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        # Store trained model state for later prediction/scoring/further training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.load_state_dict(str_to_obj(result["model_state_dict"]))

        # re-build result from serialized return value
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_result = TorchRuntimeResult()
        torch_result.job_id = job.job_id()
        torch_result.train_history = result["train_history"]["train"]
        torch_result.val_history = result["train_history"]["validation"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_result.model_state_dict = str_to_obj(result["model_state_dict"])
        torch_result.execution_time = result["execution_time"]
        return torch_result

    def predict(self, data_loader: DataLoader) -> TorchRuntimeResult:
        """Perform prediction on the passed data using the trained model
        and the Torch Infer Runtime ('torch-infer').
        All required data is serialized and sent to the server side.

        Args:
            data_loader: A PyTorch data loader object containing the inference dataset.
        Returns:
            result: A :class:`~qiskit_machine_learning.runtime.TorchRuntimeResult` object
            with the predicted results.
        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            RuntimeError: If the job execution failed.
        """
        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if self._provider is None:
            raise ValueError("The provider has not been set.")

        # Serialize inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            serial_model = obj_to_str(self._model)
            serial_data = obj_to_str(data_loader)

        # combine the settings with the serialized buffers to runtime inputs
        inputs = {
            "model": serial_model,
            "data": serial_data,
            "shots": self._shots,
            "measurement_error_mitigation": self.measurement_error_mitigation,
        }

        # define runtime options
        options = {"backend_name": self._backend.name()}

        # send job to runtime and return result
        job = self.provider.runtime.run(
            program_id="torch-infer",
            inputs=inputs,
            options=options,
        )

        # Raise a runtime error with the job id if something goes wrong
        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_result = TorchRuntimeResult()
        torch_result.job_id = job.job_id()
        torch_result.prediction = Tensor(result["prediction"])
        torch_result.execution_time = result["execution_time"]

        return torch_result

    def score(
        self, data_loader: DataLoader, score_func: Union[str, Callable]
    ) -> TorchRuntimeResult:
        """Calculate a score using the trained model and the Torch Infer Runtime ('torch-infer').
        Users can use either pre-defined score functions or their own score function.
        All required data is serialized and sent to the server side.

        Args:
            data_loader: A PyTorch data loader object containing the inference dataset.
            score_func: A string indicating one of the available scoring functions
                        ("classification" for classification, and "regression" for regression)
                        or a custom scoring function defined as
                        ``def score_func(model_output, target): -> score: float``.
        Returns:
            result: A :class:`~qiskit_machine_learning.runtime.TorchRuntimeResult` object
            with the score, a float number corresponding to the score is obtained.
        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            ValueError: If "score_func" is not "classification", "regression",
                or a custom scoring function.
            RuntimeError: If the job execution failed.
        """
        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if self._provider is None:
            raise ValueError("The provider has not been set.")

        # serialize model using pickle + dill
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            serial_model = obj_to_str(self._model)

        if score_func == "classification":
            self._score_func = _score_classification
        elif score_func == "regression":
            self._score_func = MSELoss()
        elif callable(score_func):
            self._score_func = score_func
        else:
            raise ValueError(
                '"score_func" must be a string for the available scoring functions',
                '("classification" or "regression"), or a custom scoring function.',
            )

        # serialize loss function using pickle + dill
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            serial_score_func = obj_to_str(self._score_func)

            serial_data = obj_to_str(data_loader)

        # define runtime options
        options = {"backend_name": self._backend.name()}

        # combine the settings with the serialized buffers to runtime inputs
        inputs = {
            "model": serial_model,
            "data": serial_data,
            "score_func": serial_score_func,
            "shots": self._shots,
            "measurement_error_mitigation": self.measurement_error_mitigation,
        }

        # send job to runtime and return result
        job = self.provider.runtime.run(
            program_id="torch-infer",
            inputs=inputs,
            options=options,
        )

        # Raise a runtime error with the job id if something goes wrong
        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch_result = TorchRuntimeResult()
        torch_result.job_id = job.job_id()
        torch_result.prediction = Tensor(result["prediction"])
        torch_result.score = result["score"]
        torch_result.execution_time = result["execution_time"]

        return torch_result


@deprecate_function("0.5.0")
def obj_to_str(obj: Any) -> str:
    """
    Deprecation; obj_to_str

    Encodes any object into a JSON-compatible string using dill. The intermediate
    binary data must be converted to base 64 to be able to decode into utf-8 format.

    Returns:
        The encoded string
    """
    string = base64.b64encode(dill.dumps(obj, byref=False)).decode("utf-8")
    return string


@deprecate_function("0.5.0")
def str_to_obj(string: str) -> Any:
    """
    Deprecation; str_to_obj

    Decodes a previously encoded string using dill (with an intermediate step
    converting the binary data from base 64).

    Returns:
        The decoded object
    """
    obj = dill.loads(base64.b64decode(string.encode()))
    return obj


def _score_classification(output: Tensor, target: Tensor) -> float:
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct
