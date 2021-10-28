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

"""The Qiskit Machine Learning Torch Runtime Client"""

import base64
from typing import Any, Callable, Dict, Optional, Union

import dill
from torch import Tensor as TorchTensor
from torch.nn import Module as TorchModule
from torch.nn import MSELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer as TorchOptim
from torch.utils.data import DataLoader as TorchDataloader

from qiskit.exceptions import QiskitError
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit_machine_learning.runtime.hookbase import HookBase


class TorchRuntimeResult:
    """The TorchRuntimeClient result object.

    The result objects contains the state dictionary of the trained model,
    and the training history such as the value of loss function in each epoch.
    """

    def __init__(self) -> None:
        self._job_id = None  # type: str
        self._train_history = None  # type: Dict[str, Any]
        self._val_history = None  # type: Dict[str, Any]
        self._model_state_dict = None  # type: Dict[str, Any]
        self._train_time = None  # type: float

    @property
    def job_id(self) -> str:
        """The job ID associated with the Torch runtime job."""
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: str) -> None:
        """Set the job ID associated with the Torch runtime job."""
        self._job_id = job_id

    @property
    def train_history(self) -> Dict[str, Any]:
        """The train history"""
        return self._train_history

    @train_history.setter
    def train_history(self, history: Dict[str, Any]) -> None:
        """Set the train history"""
        self._train_history = history

    @property
    def val_history(self) -> Dict[str, Any]:
        """Returns validation history"""
        return self._val_history

    @val_history.setter
    def val_history(self, history: Dict[str, Any]) -> None:
        """Sets validation history"""
        self._val_history = history

    @property
    def model_state_dict(self) -> Dict[str, Any]:
        """Returns state dictionary of trained model"""
        return self._model_state_dict

    @model_state_dict.setter
    def model_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Sets state dictionary of trained model"""
        self._model_state_dict = state_dict

    @property
    def train_time(self) -> float:
        """Returns training time"""
        return self._train_time

    @train_time.setter
    def train_time(self, time: float) -> None:
        """Sets training time"""
        self._train_time = time


class TorchRuntimeClient:
    """The Qiskit Machine Learning Torch runtime client to call the Torch runtime."""

    def __init__(
        self,
        model: TorchModule,
        optimizer: TorchOptim,
        loss_func: Union[_Loss, Callable],
        epochs: int = 100,
        shots: int = 1024,
        measurement_error_mitigation: bool = False,
        callback: Optional[Callable] = None,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        """
        Args:
            model: A PyTorch nn.Module to be trained.
            optimizer: A PyTorch optimizer for the model parameters.
            loss_func: A PyTorch-compatible loss function. Can be one of the
                official PyTorch loss functions from ``torch.nn.loss`` or a custom
                function defined by the user, as long as it follows the format
                ``def loss_func(output: TorchTensor, target: TorchTensor): -> loss: float``.
            epochs: The maximum number of training epochs. By default, 100.
            shots: The number of shots for the quantum backend. By default, 1024.
            measurement_error_mitigation: Whether or not to use measurement error mitigation.
            callback: a callback that can access the intermediate data during the optimization.
                Six parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the epoch_count, the average loss of training,
                the average loss of validating, the average forward pass time,
                the average backward pass time, the epoch time.
            provider: IBMQ provider that supports runtime services.
            backend: Selected quantum backend.
        """
        # Store settings
        self._provider = None
        self._model = None
        self._loss_func = None
        self._optimizer = None
        self._epochs = epochs
        self._shots = shots
        self._callback = callback
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
        """Set the provider. Must be a provider that supports the runtime feature."""
        try:
            _ = hasattr(provider, "runtime")
        except QiskitError:
            # pylint: disable=raise-missing-from
            raise ValueError(f"The provider {provider} does not provide a runtime environment.")

        self._provider = provider

    @property
    def model(self) -> TorchModule:
        """Return the model."""
        return self._model

    @model.setter
    def model(self, t_model: TorchModule) -> None:
        """Set the model."""
        if not isinstance(t_model, TorchModule):
            raise TypeError("The model must be an instance of torch.nn.Module")
        self._model = t_model

    @property
    def optimizer(self) -> TorchOptim:
        """Return the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optim: TorchOptim) -> None:
        """Set the optimizer."""
        if not isinstance(optim, TorchOptim):
            raise TypeError("The optimizer must be an instance of torch.optim")
        self._optimizer = optim

    @property
    def loss_func(self) -> Union[_Loss, Callable]:
        """Return the loss function."""
        return self._loss_func

    @loss_func.setter
    def loss_func(self, loss: Union[_Loss, Callable]) -> None:
        """Set the loss function."""
        if not isinstance(loss, (_Loss, Callable)):  # type: ignore
            raise TypeError(
                "The loss function must be an instance of torch.nn.Loss._Loss or Callable"
            )
        self._loss_func = loss

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

    def _wrap_torch_callback(self) -> Optional[Callable]:
        """Wraps and returns the given callback to match the signature of the runtime callback."""

        def wrapped_callback(*args):
            _, data = args  # first element is the job id
            # wrap the callback
            epoch_count = data[0]
            training_avg_loss = data[1]
            validating_avg_loss = data[2]
            avg_forward_time = data[3]
            avg_backward_time = data[4]
            epoch_time = data[5]
            self._callback(
                epoch_count,
                training_avg_loss,
                validating_avg_loss,
                avg_forward_time,
                avg_backward_time,
                epoch_time,
            )
            return self._callback(data)

        # if callback is set, return wrapped callback, else return None
        if self._callback:
            return wrapped_callback
        else:
            return None

    def obj_to_str(self, obj: Any) -> str:
        """Encodes any object into a JSON-compatible string using dill. The intermediate
        binary data must be converted to base 64 to be able to decode into utf-8 format."""
        string = base64.b64encode(dill.dumps(obj, byref=False)).decode("utf-8")
        return string

    def str_to_obj(self, string: str) -> Any:
        """Decodes a previously encoded string using dill (with an intermediate step
        converting the binary data from base 64)."""
        obj = dill.loads(base64.b64decode(string.encode()))
        return obj

    def fit(
        self,
        train_loader: TorchDataloader,
        val_loader: Optional[TorchDataloader] = None,
        hooks: Optional[HookBase] = None,
        start_epoch: int = 0,
    ) -> TorchRuntimeResult:
        """Calls the Torch Runtime train('torch-train') to train the model.

        Args:
            train_loader: A PyTorch data loader object containing the training dataset.
            val_loader: A PyTorch data loader object containing the validation dataset.
                If no validation loader is provided, the validation step will be skipped
                during training.
            hooks: List of custom hook functions to interact with the training loop.
            start_epoch: initial epoch for warm-start training.
        Returns:
            result: A :class:`~qiskit_machine_learning.runtime.TorchRuntimeResult` object
                with the trained model's state dictionary and training history data.
        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            RuntimeError: If the job execution failed.
        """

        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if self.provider is None:
            raise ValueError("The provider has not been set.")

        # serialize using dill
        serial_model = self.obj_to_str(self.model)
        serial_optim = self.obj_to_str(self.optimizer)
        serial_loss = self.obj_to_str(self.loss_func)
        serial_train_data = self.obj_to_str(train_loader)
        serial_val_data: Optional[str] = None
        if val_loader is not None:
            serial_val_data = self.obj_to_str(val_loader)
        if hooks is None:
            serial_hooks = self.obj_to_str([])
        else:
            serial_hooks = self.obj_to_str(hooks)

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
        }

        # define runtime options
        options = {"backend_name": self._backend.name()}

        import json

        with open("params.json", "w") as f:
            json.dump(inputs, f, indent=4)

        return 1

        # send job to runtime and return result
        job = self.provider.runtime.run(
            program_id="torch-train",
            inputs=inputs,
            options=options,
            callback=self._wrap_torch_callback(),
        )

        # print job ID if something goes wrong
        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        # Store trained model state for later prediction/scoring/further training
        self._model.load_state_dict(self.str_to_obj(result["model_state_dict"]))

        # re-build result from serialized return value
        torch_result = TorchRuntimeResult()
        torch_result.job_id = job.job_id()
        torch_result.train_history = result.get("train_history", {}).get("train", None)
        torch_result.val_history = result.get("train_history", {}).get("val", None)
        torch_result.model_state_dict = self.str_to_obj(result.get("model_state_dict"))
        torch_result.train_time = result.get("train_time")
        return torch_result

    def predict(self, data_loader: TorchDataloader) -> TorchTensor:
        """Calls the Torch Runtime infer ('torch-infer') for prediction purposes.

        Args:
            data_loader: A PyTorch data loader object containing the inference dataset.
        Returns:
            prediction: A PyTorch ``Tensor`` with the result of applying the model.
        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            RuntimeError: If the job execution failed.
        """

        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if self.provider is None:
            raise ValueError("The provider has not been set.")

        # Serialize inputs
        serial_model = self.obj_to_str(self._model)
        serial_data = self.obj_to_str(data_loader)

        # combine the settings with the serialized buffers to runtime inputs
        inputs = {
            "model": serial_model,
            "data": serial_data,
            "shots": self._shots,
            "measurement_error_mitigation": self.measurement_error_mitigation,
        }

        # define runtime options
        options = {"backend_name": self._backend.name()}

        import json

        with open("params.json", "w") as f:
            json.dump(inputs, f, indent=4)

        return 1

        # send job to runtime and return result
        job = self.provider.runtime.run(
            program_id="torch-infer",
            inputs=inputs,
            options=options,
            callback=self._wrap_torch_callback(),
        )

        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        out_tensor = TorchTensor(result["prediction"])
        return out_tensor

    def score(self, data_loader: TorchDataloader, score_func: Union[str, Callable]) -> float:
        """Calls the Torch Runtime infer ('torch-infer') for scoring.

        Args:
            data_loader: A PyTorch data loader object containing the inference dataset.
            score_func: A string indicating one of the available scoring functions
                        ("Classification" for binary classification, and "Regression" for regression)
                        or a custom scoring function defined as:
                        ``def score_func(model_output, target): -> score: float``.
        Returns:
            score: A metric of the model's performance.
        Raises:
            ValueError: If the backend has not yet been set.
            ValueError: If the provider has not yet been set.
            RuntimeError: If the job execution failed.
        """
        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if self.provider is None:
            raise ValueError("The provider has not been set.")

        # serialize model using pickle + dill
        serial_model = self.obj_to_str(self._model)

        if score_func == "Classification":
            self._score_func = _score_classification
        elif score_func == "Regression":
            self._score_func = MSELoss()
        elif callable(score_func):
            self._score_func = score_func
        else:
            raise ValueError("Scoring function is not provided.")

        # serialize loss function using pickle + dill
        serial_score_func = self.obj_to_str(self._score_func)
        serial_data = self.obj_to_str(data_loader)

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

        import json

        with open("params.json", "w") as f:
            json.dump(inputs, f, indent=4)

        return 1

        # send job to runtime and return result
        job = self.provider.runtime.run(
            program_id="torch-infer",
            inputs=inputs,
            options=options,
            callback=self._wrap_torch_callback(),
        )

        try:
            result = job.result()
        except Exception as exc:
            raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        return result["score"]


def _score_classification(output: TorchTensor, target: TorchTensor) -> float:
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct
