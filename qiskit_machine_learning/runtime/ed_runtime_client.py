# This code is part of qiskit-runtime.
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

"""The Effective Dimension Runtime Program Client."""
from typing import List, Union, Callable, Optional, Any, Dict
from qiskit.exceptions import QiskitError
from qiskit.providers import Provider
from qiskit.providers.backend import Backend
from qiskit_ibm_runtime import IBMRuntimeService as Service

from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import NeuralNetwork

import dill, base64
import numpy as np
import time

class EffDimRuntimeClient:
    """Torch Runtime Program Client"""

    def __init__(
        self,
        qnn: NeuralNetwork,
        shots: int = 1024,
        mock_runtime: bool = False,
        measurement_error_mitigation: bool = False,
        callback: Optional[Callable] = None,
        service: Optional[Service] = None,
        backend: Optional[Backend] = None,
        program_id = None
    ) -> None:
        """
        Args:
            ......

            shots: The number of shots for the quantum backend. By default, 1024.
            mock_runtime: Flag used for debugging purposes. If ``True``, the runtime
                program will be executed in the local environment (requires having
                access to the runtime code).
            measurement_error_mitigation: Whether or not to use measurement error mitigation.
            callback: If provided, callback function that will be executed.
            provider: IBMQ provider that supports runtime services.
            backend: Selected quantum backend.
        """

        # Store settings
        self._service = None
        self._backend = backend
        # TO-DO: getters, setters. Make private variables
        self._shots = shots
        self._callback = callback
        self._mock_runtime = mock_runtime
        self._measurement_error_mitigation = measurement_error_mitigation

        self.program_id = program_id
        # Use setter to check for valid inputs
        if service is not None:
            self.service = service

        if self._mock_runtime:
            # This section is for debugging purposes and not general at all.
            # Might make sense to remove it in the final version of the runtime code.
            import sys

            sys.path.append("../..")
            from qiskit_machine_learning.runtime.mock_run_utils.user_messenger import UserMessenger

            self.user_messenger = UserMessenger()


        # if feat_map.num_qubits != ansatz.num_qubits:
        #     print("NUM QUBITS ERROR") # DO BETTER
        #
        # qc = QuantumCircuit(feat_map.num_qubits)
        # qc.append(feat_map, range(feat_map.num_qubits))
        # qc.append(ansatz, range(ansatz.num_qubits))
        #
        # # parity maps bitstrings to 0 or 1
        # def parity(x):
        #     return "{:b}".format(x).count("1") % 2
        #
        # # construct QNN. DO NOT SET QUANTUM INSTANCE HERE (it has to be done inside the runtime)
        # qnn = CircuitQNN(
        #     qc,
        #     input_params=feat_map.parameters,
        #     weight_params=ansatz.parameters,
        #     interpret=parity,
        #     output_shape=2,
        #     sparse=False
        # )

        self._qnn = qnn
        self.d = qnn.num_weights

    @property
    def service(self) -> Optional[Service]:
        """Return the service."""
        return self._service

    @service.setter
    def service(self, service: Service) -> None:
        """Set the service. Must be a service that supports the runtime feature."""
        # try:
        #     _ = hasattr(service, "runtime")
        # except QiskitError:
        #     # pylint: disable=raise-missing-from
        #     raise ValueError(f"The service {service} does not provide a runtime environment.")

        self._service = service

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
            return self._callback(data)

        # if callback is set, return wrapped callback, else return None
        if self._callback:
            return wrapped_callback
        else:
            return None

    # TODO: encoder/decoder class (or it can be left like this)
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

    def compute_eff_dim(self,
                n: Union[int, List],
                num_inputs: Optional[int] = 0,
                num_thetas: Optional[int] = 0,
                inputs: Optional[List] = None,
                thetas: Optional[List] = None,
                ):

        """
        Compute the effective dimension.
        :param f_hat: ndarray
        :param n: list, used to represent number of data samples available as per the effective dimension calc
        :return: list, effective dimension for each n
        """

        if self._backend is None:
            raise ValueError("The backend has not been set.")

        if not self._mock_runtime and self.service is None:
            raise ValueError("The service has not been set.")

        # No need for self.num_inputs/num_thetas
        if thetas is not None:
            self.thetas = thetas
        elif num_thetas is not None:
            self.thetas = np.random.uniform(0, 1, size=(num_thetas, self.d))
        if inputs is not None:
            self.inputs = inputs
        elif num_inputs is not None:
            self.inputs = np.random.normal(0, 1, size=(num_inputs, self._qnn.num_inputs))

        # serialize using dill
        serial_inputs = self.inputs.tolist()
        serial_thetas = self.thetas.tolist()
        serial_qnn = self.obj_to_str(self._qnn)
        # USE QPY HERE? # REMINDER: qiskit-machine-learning versions have to match

        # define runtime options
        options = {"backend_name": self._backend.name}

        if self._mock_runtime:
            # combine the settings with the serialized buffers to runtime inputs
            inputs = {
                "qnn": serial_qnn,
                "inputs": serial_inputs,
                "thetas": serial_thetas,
                "shots": self._shots,
                "n": n,
                "measurement_error_mitigation": self.measurement_error_mitigation,
                "options": options,
                "callback": self._wrap_torch_callback(),
            }
            from qiskit_machine_learning.runtime.ed_runtime_program import main

            # run mock runtime
            result = main(
                inputs=inputs, user_messenger=self.user_messenger, backend=self._backend
            )

        else:
            # combine the settings with the serialized buffers to runtime inputs
            inputs = {
                "qnn": serial_qnn,
                "inputs": serial_inputs,
                "thetas": serial_thetas,
                "shots": self._shots,
                "measurement_error_mitigation": self.measurement_error_mitigation,
            }

            # send job to runtime and return result
            job = self.service.run(
                program_id=self.program_id,
                inputs=inputs,
                options=options,
                callback=self._wrap_torch_callback(),
            )

            # print job ID if something goes wrong
            try:
                result = job.result()
            except Exception as exc:
                raise RuntimeError(f"The job {job.job_id()} failed unexpectedly.") from exc

        return result






