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
from __future__ import annotations

import io
import hashlib
from qiskit import qpy, QuantumCircuit


def circuit_cache_key(circ: QuantumCircuit) -> str:
    """
    Generate a deterministic, stable cache key for a QuantumCircuit using QPY serialization.

    This function produces a reproducible key by serializing the given circuit to its
    QPY (Quantum Program) binary representation in memory, without writing any files
    to disk. The QPY format is Qiskit’s canonical and version-stable representation of
    circuits, preserving structure, parameters, and metadata. By hashing the resulting
    bytes, we obtain a unique fingerprint that changes only if the circuit’s logical
    content changes.

    The implementation mirrors the behavior of :func:`qiskit.qpy.dump`, which normally
    writes to a file object. Here, instead of saving to disk (e.g., ``with open('file.qpy', 'wb')``),
    we direct the output to an in-memory :class:`io.BytesIO` buffer that is discarded after use.

    Parameters
    ----------
    circ : QuantumCircuit
        The circuit to serialize and hash.

    Returns
    -------
    str
        A deterministic hexadecimal digest (SHA-256) of the circuit’s QPY byte representation.
        This can safely be used as a dictionary or cache key.

    Notes
    -----
    - Using QPY ensures compatibility across Qiskit versions and Python sessions.
    - Unlike Python’s built-in ``hash()``, the SHA-256 digest is stable across runs.
    - This approach avoids file I/O entirely, as serialization happens in memory.

    Example
    -------

    .. code-block:: python

        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        key = circuit_cache_key(qc)
        print(key)
        # Output: '5e341a63f4c6a9d17a3d72b1c07d2ac4b8e9a7a1fbb9b7d93f6d6d2f0b59a6f2'

    """
    buffer = io.BytesIO()
    # QPY expects a list of programs (can be a single circuit or list)
    qpy.dump([circ], buffer)
    qpy_bytes = buffer.getvalue()
    return hashlib.sha256(qpy_bytes).hexdigest()
