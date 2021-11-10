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
import base64
import dill
from qiskit.providers import Provider


class FakeQKTJob:
    """A fake job for unit tests."""

    def __init__(self, serial_qkernel):
        self._quantum_kernel = self.str_to_obj(serial_qkernel)

    def obj_to_str(self, obj: Any) -> str:
        """
        Encodes any object into a JSON-compatible string using dill. The intermediate
        binary data must be converted to base 64 to be able to decode into utf-8 format.
        """
        string = base64.b64encode(dill.dumps(obj, byref=False)).decode("utf-8")
        return string

    def str_to_obj(self, string: str) -> Any:
        """Decodes a previously encoded string using dill (with an intermediate step
        converting the binary data from base 64)."""
        obj = dill.loads(base64.b64decode(string.encode()))
        return obj

    def result(self) -> Dict[str, Any]:
        """Return a QKT program result."""
        # pylint: disable=no-member
        num_params = len(self._quantum_kernel.user_parameters)
        # pylint: enable=no-member
        serialized_result = {
            "quantum_kernel": self.obj_to_str(self._quantum_kernel),
            "optimal_point": [0.1] * num_params,
        }

        return serialized_result

    def job_id(self) -> str:
        """Return a fake job ID."""
        return "c2985khdm6upobbnmll0"


class FakeQKTRuntime:
    """A fake QKT runtime for unit tests."""

    def run(self, program_id, inputs, options, callback=None):
        """Run the fake program. Checks the input types."""

        if program_id != "quantum-kernel-training":
            raise ValueError("program_id is not quantum-kernel-training.")

        allowed_inputs = {
            "quantum_kernel": str,
            "data": str,
            "labels": str,
            "optimizer": str,
            "shots": int,
            "measurement_error_mitigation": bool,
            "initial_point": str,
        }

        for arg, value in inputs.items():
            if not isinstance(value, allowed_inputs[arg]):
                raise ValueError(f"{arg} does not have the right type: {allowed_inputs[arg]}")

        allowed_options = {"backend_name": str, "initial_layout": str}
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
