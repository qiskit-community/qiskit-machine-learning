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
from typing import Any, Callable, Dict, Optional, Union, List

import dill

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.exceptions import QiskitError
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit_machine_learning.runtime.hookbase import HookBase

try:
    from torch import Tensor
    from torch.nn import Module, MSELoss
    from torch.nn.modules.loss import _Loss
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader
except ImportError:

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

        def __init__(self) -> None:
            raise MissingOptionalLibraryError(
                libname="Pytorch",
                name="TorchConnector",
                pip_install="pip install 'qiskit-machine-learning[torch]'",
            )


class TorchRuntimeResult:
    """The TorchRuntimeClient result object.

    The result objects contains the state dictionary of the trained model,
    and the training history such as the value of loss function in each epoch.
    """

    def __init__(self) -> None:
        self._job_id: Optional[str] = None
        self._train_history: Optional[Dict[str, Any]] = None
        self._val_history: Optional[Dict[str, Any]] = None
        self._model_state_dict: Optional[Dict[str, Any]] = None
        self._train_time: Optional[float] = None

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
        model: Module,
        optimizer: Optimizer,
        loss_func: Union[_Loss, Callable],
        epochs: int = 10,
        shots: int = 1024,
        measurement_error_mitigation: bool = False,
        provider: Optional[Provider] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        """
        Args:
            model: A PyTorch nn.Module to be trained.
            optimizer: A PyTorch optimizer for the model parameters.
            loss_func: A PyTorch-compatible loss function. Can be one of the
                official PyTorch loss functions from ``torch.nn.loss`` or a custom
                function defined by the user.
            epochs: The maximum number of training epochs. By default, 10.
            shots: The number of shots for the quantum backend. By default, 1024.
            measurement_error_mitigation: Whether or not to use measurement error mitigation.
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
    def model(self) -> Module:
        """Return the model."""
        return self._model

    @model.setter
    def model(self, t_model: Module) -> None:
        """Set the model."""
        if not isinstance(t_model, Module):
            raise TypeError("The model must be an instance of torch.nn.Module")
        self._model = t_model

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optim: Optimizer) -> None:
        """Set the optimizer."""
        if not isinstance(optim, Optimizer):
            raise TypeError("The optimizer must be an instance of torch.optim")
        self._optimizer = optim

    @property
    def loss_func(self) -> Union[_Loss, Callable]:
        """Return the loss function."""
        return self._loss_func

    @loss_func.setter
    def loss_func(self, loss: Union[_Loss, Callable]) -> None:
        """Set the loss function."""
        if isinstance(loss, _Loss) or callable(loss):
            self._loss_func = loss
        else:
            raise TypeError(
                "The loss function must be an instance of torch.nn.Loss._Loss or Callable"
            )

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
        """Train the model using the Torch Runtime train('torch-train').
        All necessary data is serialized and it's sent to the server side.
        After training, model's parameter is updated and TorchRuntimeResult is returned for the result.

        Args:
            train_loader: A PyTorch data loader object containing the training dataset.
            val_loader: A PyTorch data loader object containing the validation dataset.
                If no validation loader is provided, the validation step will be skipped
                during training.
            hooks: a hook class of a List of hook classes to interact with the training loop.
            start_epoch: initial epoch for warm-start training.
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
        serial_model = obj_to_str(self.model)
        serial_optim = obj_to_str(self.optimizer)
        serial_loss = obj_to_str(self.loss_func)
        serial_train_data = obj_to_str(train_loader)
        serial_val_data: Optional[str] = None
        # check hooks
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
        self._model.load_state_dict(str_to_obj(result["model_state_dict"]))

        # re-build result from serialized return value
        torch_result = TorchRuntimeResult()
        torch_result.job_id = job.job_id()
        torch_result.train_history = result.get("train_history", {}).get("train", None)
        torch_result.val_history = result.get("train_history", {}).get("val", None)
        torch_result.model_state_dict = str_to_obj(result.get("model_state_dict"))
        torch_result.train_time = result.get("train_time")
        return torch_result

    def predict(self, data_loader: DataLoader) -> Tensor:
        """Predict the result using the trained model and the Torch Runtime infer ('torch-infer').
        All necessary data is serialized and it's sent to the server side.
        After predicting, a Tensor corresponding to the predicted result of the input data is returned.

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

        if self._provider is None:
            raise ValueError("The provider has not been set.")

        # Serialize inputs
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

        return Tensor(result["prediction"])

    def score(self, data_loader: DataLoader, score_func: Union[str, Callable]) -> float:
        """Calculate the score using the trained model and the Torch Runtime infer ('torch-infer').
        Users can use either pre-defined score functions or their own score function.
        All necessary data is serialized and it's sent to the server side.
        After calculating the score, a float number corresponding to the score is returned.

        Args:
            data_loader: A PyTorch data loader object containing the inference dataset.
            score_func: A string indicating one of the available scoring functions
                        ("classification" for classification, and "regression" for regression)
                        or a custom scoring function defined as:
                        ``def score_func(model_output, target): -> score: float``.
        Returns:
            score: A metric of the model's performance.
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

        return result["score"]


def obj_to_str(obj: Any) -> str:
    """
    Encodes any object into a JSON-compatible string using dill. The intermediate
    binary data must be converted to base 64 to be able to decode into utf-8 format.

    Returns:
        The encoded string
    """
    string = base64.b64encode(dill.dumps(obj, byref=False)).decode("utf-8")
    return string


def str_to_obj(string: str) -> Any:
    """
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
