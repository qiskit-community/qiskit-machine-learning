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

"""Fake runtime provider and quantum-kernel-trainer runtime."""

from typing import Dict, Any, Iterable
from qiskit.providers import Provider
from qiskit.algorithms.optimizers import Optimizer

from qiskit_machine_learning.kernels import QuantumKernel


class FakeQKTJob:
    """A fake job for unit tests."""

    def __init__(self, quantum_kernel):
        self._quantum_kernel = quantum_kernel

    def result(self) -> Dict[str, Any]:
        """Return a quantum-kernel-trainer program result."""
        # pylint: disable=no-member
        num_params = len(self._quantum_kernel.user_parameters)
        # pylint: enable=no-member
        serialized_result = {
            "quantum_kernel": self._quantum_kernel,
            "optimal_point": [0.1] * num_params,
        }

        return serialized_result

    def job_id(self) -> str:
        """Return a fake job ID."""
        return "c2985khdm6upobbnmll0"


class FakeQKTRuntime:
    """A fake quantum-kernel-trainer runtime for unit tests."""

    def run(self, program_id, inputs, options, callback=None):
        """Run the fake program. Checks the input types."""

        if program_id != "quantum-kernel-training":
            raise ValueError("program_id is not quantum-kernel-training.")

        allowed_inputs = {
            "quantum_kernel": QuantumKernel,
            "data": Iterable,
            "labels": Iterable,
            "optimizer": Optimizer,
            "shots": int,
            "measurement_error_mitigation": bool,
            "initial_point": Iterable,
        }

        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f"{arg} does not have the right type: {type(value)}")

        allowed_options = {"backend_name": str, "initial_layout": Iterable}

        for arg, value in options.items():
            if not isinstance(value, allowed_options[arg]):
                raise ValueError(f"{arg} does not have the right type: {type(value)}")

        if callback is not None:
            try:
                fake_job_id = "c2985khdm6upobbnmll0"
                fake_data = [1, 0.9, 0.8, 10, 60, 70]
                _ = callback(fake_job_id, fake_data)
            except Exception as exc:
                raise ValueError("Callback failed") from exc

        return FakeQKTJob(inputs["quantum_kernel"])


class FakeQKTRuntimeProvider(Provider):
    """A fake runtime provider for unit tests."""

    def has_service(self, service):
        """Check if a service is available."""
        if service == "runtime":
            return True
        return False

    @property
    def runtime(self) -> FakeQKTRuntime:
        """Return the runtime."""
        return FakeQKTRuntime()
