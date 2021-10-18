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

"""Fake runtime provider and VQE runtime."""

from typing import Dict, Any
from qiskit.providers import Provider
from qiskit_machine_learning.runtime import TorchProgramResult


class FakeTorchTrainerJob:
    """A fake job for unit tests."""

    def result(self) -> Dict[str, Any]:
        """Return a Torch program result."""
        result = TorchProgramResult()
        serialized_result = {
            "model_state_dict": result.model_state_dict,
            "train_history": result.train_history,
            "train_time": result.train_time
        }
        return serialized_result

    def job_id(self) -> str:
        """Return a fake job ID."""
        return "c2985khdm6upobbnmll0"

class FakeTorchInferJob:
    """A fake job for unit tests."""

    def result(self) -> Dict[str, Any]:
        """Return a Torch program result."""
        serialized_result = {
            "prediction": [],
            "time": 1,
        }
        return serialized_result

    def job_id(self) -> str:
        """Return a fake job ID."""
        return "c2985khdm6upobbnmll0"

class FakeTorchRuntimeTrainer:
    """A fake Torch runtime for unit tests."""

    def run(self, program_id, inputs, options, callback=None):
        """Run the fake program. Checks the input types."""

        if program_id != "torch-train":
            raise ValueError("program_id is not torch-train.")

        allowed_inputs = {
            "model": str,
            "optimizer": str,
            "loss_func": str,
            "train_data": str,
            "val_data": str,
            "shots": int,
            "measurement_error_mitigation": bool,
            "epochs": int,
            "start_epoch": int,
            "hooks": str
        }
        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        allowed_options = {"backend_name": str}
        for arg, value in options.items():
            if not isinstance(value, allowed_options[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        if callback is not None:
            try:
                fake_job_id = "c2985khdm6upobbnmll0"
                fake_data = [1, 0.9, 0.8, 10, 60, 70]
                _ = callback(fake_job_id, fake_data)
            except Exception as exc:
                raise ValueError("Callback failed") from exc

        return FakeTorchTrainerJob()

class FakeTorchRuntimeInfer:
    """A fake Torch runtime for unit tests."""

    def run(self, program_id, inputs, options, callback=None):
        """Run the fake program. Checks the input types."""

        if program_id != "torch-infer":
            raise ValueError("program_id is not torch-infer.")

        allowed_inputs = {
            "model": str,
            "data": str,
            "shots": int,
            "measurement_error_mitigation": bool,
        }
        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        allowed_options = {"backend_name": str}
        for arg, value in options.items():
            if not isinstance(value, allowed_options[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")
        return FakeTorchInferJob()

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
