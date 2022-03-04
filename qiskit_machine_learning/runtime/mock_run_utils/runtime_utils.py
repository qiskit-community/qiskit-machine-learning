# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=method-hidden
# pylint: disable=too-many-return-statements

"""Utility functions for the runtime service."""

import json
from typing import Any, Callable, Dict
import base64
import io
import zlib
import inspect
import importlib
import warnings

import numpy as np

try:
    import scipy.sparse

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from qiskit.result import Result
from qiskit.circuit import QuantumCircuit, qpy_serialization
from qiskit.circuit import ParameterExpression, Instruction
from qiskit.circuit.library import BlueprintCircuit


def _serialize_and_encode(
    data: Any, serializer: Callable, compress: bool = True, **kwargs: Any
) -> str:
    """Serialize the input data and return the encoded string.
    Args:
        data: Data to be serialized.
        serializer: Function used to serialize data.
        compress: Whether to compress the serialized data.
        kwargs: Keyword arguments to pass to the serializer.
    Returns:
        String representation.
    """
    buff = io.BytesIO()
    serializer(buff, data, **kwargs)
    buff.seek(0)
    serialized_data = buff.read()
    buff.close()
    if compress:
        serialized_data = zlib.compress(serialized_data)
    return base64.standard_b64encode(serialized_data).decode("utf-8")


def _decode_and_deserialize(data: str, deserializer: Callable, decompress: bool = True) -> Any:
    """Decode and deserialize input data.
    Args:
        data: Data to be deserialized.
        deserializer: Function used to deserialize data.
        decompress: Whether to decompress.
    Returns:
        Deserialized data.
    """
    buff = io.BytesIO()
    decoded = base64.standard_b64decode(data)
    if decompress:
        decoded = zlib.decompress(decoded)
    buff.write(decoded)
    buff.seek(0)
    orig = deserializer(buff)
    buff.close()
    return orig


def deserialize_from_settings(mod_name: str, class_name: str, settings: Dict) -> Any:
    """Deserialize an object from its settings.
    Args:
        mod_name: Name of the module.
        class_name: Name of the class.
        settings: Object settings.
    Returns:
        Deserialized object.
    Raises:
        ValueError: If unable to find the class.
    """
    mod = importlib.import_module(mod_name)
    for name, clz in inspect.getmembers(mod, inspect.isclass):
        if name == class_name:
            return clz(**settings)
    raise ValueError(f"Unable to find class {class_name} in module {mod_name}")


class RuntimeEncoder(json.JSONEncoder):
    """JSON Encoder used by runtime service."""

    def default(self, obj: Any) -> Any:  # pylint: disable=arguments-differ
        if isinstance(obj, complex):
            return {"__type__": "complex", "__value__": [obj.real, obj.imag]}
        if isinstance(obj, np.ndarray):
            value = _serialize_and_encode(obj, np.save, allow_pickle=False)
            return {"__type__": "ndarray", "__value__": value}
        if isinstance(obj, set):
            return {"__type__": "set", "__value__": list(obj)}
        if isinstance(obj, Result):
            return {"__type__": "Result", "__value__": obj.to_dict()}
        if hasattr(obj, "to_json"):
            return {"__type__": "to_json", "__value__": obj.to_json()}
        if isinstance(obj, QuantumCircuit):
            # TODO Remove the decompose when terra 6713 is released.
            if isinstance(obj, BlueprintCircuit):
                obj = obj.decompose()
            value = _serialize_and_encode(
                data=obj, serializer=lambda buff, data: qpy_serialization.dump(data, buff)
            )
            return {"__type__": "QuantumCircuit", "__value__": value}
        if isinstance(obj, ParameterExpression):
            value = _serialize_and_encode(
                data=obj,
                serializer=qpy_serialization._write_parameter_expression,
                compress=False,
            )
            return {"__type__": "ParameterExpression", "__value__": value}
        if isinstance(obj, Instruction):
            value = _serialize_and_encode(
                data=obj, serializer=qpy_serialization._write_instruction, compress=False
            )
            return {"__type__": "Instruction", "__value__": value}
        if hasattr(obj, "settings"):
            return {
                "__type__": "settings",
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "__value__": obj.settings,
            }
        if callable(obj):
            warnings.warn(f"Callable {obj} is not JSON serializable and will be set to None.")
            return None
        if HAS_SCIPY and isinstance(obj, scipy.sparse.spmatrix):
            value = _serialize_and_encode(obj, scipy.sparse.save_npz, compress=False)
            return {"__type__": "spmatrix", "__value__": value}
        return super().default(obj)


class RuntimeDecoder(json.JSONDecoder):
    """JSON Decoder used by runtime service."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Any) -> Any:
        """Called to decode object."""
        if "__type__" in obj:
            obj_type = obj["__type__"]
            obj_val = obj["__value__"]

            if obj_type == "complex":
                return obj_val[0] + 1j * obj_val[1]
            if obj_type == "ndarray":
                return _decode_and_deserialize(obj_val, np.load)
            if obj_type == "set":
                return set(obj_val)
            if obj_type == "QuantumCircuit":
                return _decode_and_deserialize(obj_val, qpy_serialization.load)[0]
            if obj_type == "ParameterExpression":
                return _decode_and_deserialize(
                    obj_val, qpy_serialization._read_parameter_expression, False
                )
            if obj_type == "Instruction":
                return _decode_and_deserialize(obj_val, qpy_serialization._read_instruction, False)
            if obj_type == "settings":
                return deserialize_from_settings(
                    mod_name=obj["__module__"], class_name=obj["__class__"], settings=obj_val
                )
            if obj_type == "Result":
                return Result.from_dict(obj_val)
            if obj_type == "spmatrix":
                return _decode_and_deserialize(obj_val, scipy.sparse.load_npz, False)
            if obj_type == "to_json":
                return obj_val
        return obj
