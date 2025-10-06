# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Helper function(s) to hash circuits and speed up pass managers."""

import json
import hashlib
from typing import Any
import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter, ParameterExpression


def _param_to_jsonable(p: Any) -> Any:
    """Cast a gate parameter into a JSON-serializable, deterministic form."""
    # ParameterExpression (covers Parameter too, but treat Parameter explicitly first)
    if isinstance(p, Parameter):
        return {"type": "Parameter", "name": p.name}
    if isinstance(p, ParameterExpression):
        # Use string expression + sorted parameter names for determinism
        names = sorted(par.name for par in p.parameters)
        return {"type": "ParameterExpression", "expr": str(p), "params": names}

    # Numpy scalars
    if isinstance(p, np.number):
        return float(p)

    # Plain numbers
    if isinstance(p, (int, float)):
        return float(p)

    # Complex numbers
    if isinstance(p, complex):
        return {"type": "complex", "re": float(p.real), "im": float(p.imag)}

    # Fallback: stable string form
    return {"type": type(p).__name__, "repr": repr(p)}


def circuit_cache_key(circ: QuantumCircuit) -> str:
    """
    Deterministic structural hash for a circuit using a canonical JSON encoding.

    Encodes:
      - num_qubits / num_clbits
      - global_phase (if any)
      - operations as a list of {name, qinds, cinds, params}
        where qinds/cinds are indices into circ.qubits / circ.clbits,
        and params are serialized via `_param_to_jsonable`.

    Notes:
      - This is lighter-weight than QPY but less exhaustive (e.g., calibrations/metadata
        are not included). If you need a *fully robust* fingerprint, prefer the QPY-based
        approach we discussed earlier.
    """
    q_index = {q: i for i, q in enumerate(circ.qubits)}
    c_index = {c: i for i, c in enumerate(circ.clbits)}

    ops = []
    for inst in circ.data:
        name = inst.operation.name
        qinds = [q_index[q] for q in inst.qubits]
        cinds = [c_index[c] for c in inst.clbits]
        params = [_param_to_jsonable(p) for p in getattr(inst.operation, "params", ())]
        ops.append({"name": name, "q": qinds, "c": cinds, "params": params})

    meta = {
        "num_qubits": circ.num_qubits,
        "num_clbits": circ.num_clbits,
        # Add below if you want them to affect the key:
        # "global_phase": float(circ.global_phase) if circ.global_phase else 0.0,1
        # "name": circ.name,
        # "metadata": circ.metadata,   # must be JSON-serializable if enabled
    }

    payload = {"meta": meta, "ops": ops}
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
