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

"""Fake runtime provider and Torch runtime."""

from typing import Dict, Any

from qiskit.providers import Provider
from qiskit_machine_learning.runtime import str_to_obj, obj_to_str


class FakeTorchTrainerJob:
    """A fake job for unit tests."""

    def __init__(self, serialized_model, val_data):
        model = str_to_obj(serialized_model)
        self._model_state_dict = obj_to_str(model.state_dict())
        self._validate = val_data is not None

    def result(self) -> Dict[str, Any]:
        """Return a Torch program result."""
        train_history = {}
        train_history["train"] = [
            {
                "epoch": 0,
                "loss": 0.1,
                "forward_time": 0.1,
                "backward_time": 0.1,
                "epoch_time": 0.2,
            }
        ]
        if self._validate:
            train_history["validation"] = [
                {
                    "epoch": 0,
                    "loss": 0.2,
                    "forward_time": 0.2,
                    "backward_time": 0.2,
                    "epoch_time": 0.4,
                }
            ]
        else:
            train_history["validation"] = []
        serialized_result = {
            "model_state_dict": self._model_state_dict,
            "train_history": train_history,
            "execution_time": 0.2,
        }

        return serialized_result

    def job_id(self) -> str:
        """Return a fake job ID."""
        return "c2985khdm6upobbnmll0"


class FakeTorchInferJob:
    """A fake job for unit tests."""

    def __init__(self, is_score):
        self._is_score = is_score

    def result(self) -> Dict[str, Any]:
        """Return a Torch program result."""
        serialized_result = {
            "prediction": [1],
            "execution_time": 0.1,
        }
        if self._is_score:
            serialized_result["score"] = 1

        return serialized_result

    def job_id(self) -> str:
        """Return a fake job ID."""
        return "c2985khdm6upobbnmll0"


class FakeTorchRuntimeTrainer:
    """A fake Torch runtime for unit tests."""

    def run(self, program_id, inputs, options):
        """Run the fake program. Checks the input types."""

        if program_id != "torch-train":
            raise ValueError("program_id is not torch-train.")

        allowed_inputs = {
            "model": str,
            "optimizer": str,
            "loss_func": str,
            "train_data": str,
            "val_data": (str, type(None)),
            "shots": int,
            "measurement_error_mitigation": bool,
            "epochs": int,
            "start_epoch": int,
            "hooks": (str),
            "seed": (int, type(None)),
        }
        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        allowed_options = {"backend_name": str}
        for arg, value in options.items():
            if not isinstance(value, allowed_options[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        return FakeTorchTrainerJob(inputs["model"], inputs["val_data"])


class FakeTorchRuntimeInfer:
    """A fake Torch runtime for unit tests."""

    def run(self, program_id, inputs, options):
        """Run the fake program. Checks the input types."""

        if program_id != "torch-infer":
            raise ValueError("program_id is not torch-infer.")

        allowed_inputs = {
            "model": str,
            "data": str,
            "shots": int,
            "measurement_error_mitigation": bool,
        }
        if "score_func" in inputs:
            allowed_inputs["score_func"] = str

        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        allowed_options = {"backend_name": str}
        for arg, value in options.items():
            if not isinstance(value, allowed_options[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")
        return FakeTorchInferJob(is_score="score_func" in inputs)


class FakeTorchTrainerRuntimeProvider(Provider):
    """A fake runtime provider for unit tests."""

    def has_service(self, service):
        """Check if a service is available."""
        if service == "runtime":
            return True
        return False

    @property
    def runtime(self) -> FakeTorchRuntimeTrainer:
        """Return the runtime for a torch trainer."""
        return FakeTorchRuntimeTrainer()


class FakeTorchInferRuntimeProvider(Provider):
    """A fake runtime provider for unit tests."""

    def has_service(self, service):
        """Check if a service is available."""
        if service == "runtime":
            return True
        return False

    @property
    def runtime(self) -> FakeTorchRuntimeInfer:
        """Return the runtime for a torch infer."""
        return FakeTorchRuntimeInfer()
