# This code is part of qiskit-runtime.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Runtime program for (Hybrid) QNN training using PyTorch."""

from typing import Optional, Callable, Tuple, List, Dict, Any
import time
import dill, base64, weakref
import numpy as np

from qiskit.utils import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit_machine_learning.algorithms.effective_dimension import EffectiveDimension

class Publisher:
    """Class used to publish interim results."""

    def __init__(self, messenger):
        self._messenger = messenger

    def callback(self, *args, **kwargs):
        text = list(args)
        for k, v in kwargs.items():
            text.append({k: v})
        self._messenger.publish(text)

def obj_to_str(obj):
    string = base64.b64encode(dill.dumps(obj, byref=False)).decode("utf-8")
    return string

def str_to_obj(string):
    obj = dill.loads(base64.b64decode(string.encode()))
    return obj


def main(backend, user_messenger=None, **kwargs):
    """Entry function."""
    # Debug msg
    user_messenger.publish("Beginning runtime program")
    # If 'inputs' is a key of kwargs, we are on mock runtime mode
    mock_inputs = kwargs.get("inputs", None)
    if mock_inputs is not None:
        # This is a little trick to be able to debug in mock runtime
        # mode. If using real runtime, the inputs are the kwargs. If
        # using mock runtime, the kwargs are a dictionary with an
        # entry called "inputs".
        user_messenger.publish("Mock runtime mode")
        kwargs = kwargs["inputs"]

    # Define mandatory arguments
    # mandatory = {"model", "optimizer", "loss_func", "train_data"}
    # missing = mandatory - set(kwargs.keys())
    # if len(missing) > 0:
    #     raise ValueError(f"The following mandatory arguments are missing: {missing}.")

    # Set up publisher
    publisher = Publisher(user_messenger)

    # Get inputs and deserialize
    qnn = str_to_obj(kwargs["qnn"])
    inputs = kwargs.get("inputs", None)
    thetas = kwargs.get("thetas", None)
    n = kwargs.get("n", None)

    shots = kwargs.get("shots", 1024)
    measurement_error_mitigation = kwargs.get('measurement_error_mitigation', False)

    # Set quantum instance
    if measurement_error_mitigation:
        _quantum_instance = QuantumInstance(backend,
                                            shots=shots,
                                            measurement_error_mitigation_shots=shots,
                                            measurement_error_mitigation_cls=CompleteMeasFitter)
    else:
        _quantum_instance = QuantumInstance(backend,
                                            shots=shots)

    # Set quantum instance for QNN
    qnn.quantum_instance = _quantum_instance

    ed = EffectiveDimension(qnn, thetas=thetas, inputs=inputs)

    result = ed.eff_dim(n)

    serialized_result = {
        "eff_dim": result[0],
        "time": result[1]
    }
    # This line works as the "return" statement in the real
    # runtime environment.
    user_messenger.publish(serialized_result, final=True)

    # If mock mode, return explicitly.
    if mock_inputs is not None:
        return serialized_result


if __name__ == "__main__":
    import sys, traceback, json
    from qiskit.providers.ibmq.runtime.utils import RuntimeDecoder
    from qiskit import Aer

    # The code currently uses Aer instead of runtime provider
    _backend = Aer.get_backend("qasm_simulator")
    user_params = {}
    if len(sys.argv) > 1:
        # If there are user parameters.
        user_params = json.loads(sys.argv[1], cls=RuntimeDecoder)
    try:
        main(_backend, **user_params)
    except Exception:
        print(traceback.format_exc())
