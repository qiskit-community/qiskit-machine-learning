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
"""Qiskit Machine Learning Sampler"""
from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict
from typing import Iterable, Mapping, Any
from types import SimpleNamespace

import numpy as np
from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import (
    StatevectorSampler,
    DataBin,
    PrimitiveJob,
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub


class QMLSampler(StatevectorSampler):
    """
    V2 sampler with two modes:
      - shots=None (default): exact mode, no sampling. Returns deterministic probabilities.
      - shots=int           : sampling mode, delegate to StatevectorSampler with given default_shots.
    """

    def __init__(self, *, shots: int | None = None, **kwargs):
        """Statevector-based sampler supporting exact (analytic) and sampling modes.

        Args:
            shots (int | None): Number of shots for sampling mode. If ``None``, run in exact mode.
            **kwargs: Additional arguments forwarded to StatevectorSampler.

        Returns:
            QMLSampler: Configured sampler instance.
        """
        self._exact_mode = shots is None
        if self._exact_mode:
            super().__init__(**kwargs)
        else:
            super().__init__(default_shots=int(shots), **kwargs)

        parent_opts = object.__getattribute__(self, "__dict__").get("options", None)
        base = _options_to_dict(parent_opts)
        merged = dict(base)
        merged.setdefault("default_shots", shots)
        self.options = _OptionsNS(**merged)

    def run(
        self,
        pubs: Iterable[SamplerPubLike],
        *,
        shots: int | None = None,
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        """Run the sampler on PUBs.

        Args:
            pubs (Iterable[SamplerPubLike]): Publications to evaluate.
            shots (int | None): Optional override for number of shots.

        Returns:
            PrimitiveJob[PrimitiveResult[SamplerPubResult]]: Job executing the sampler.
        """
        if not self._exact_mode:
            return super().run(pubs, shots=shots)

        # Exact mode: compute probabilities from statevector, no sampling.
        coerced = [SamplerPub.coerce(pub, shots=1) for pub in pubs]  # satisfy validation
        job = PrimitiveJob(self._run_exact, coerced)
        job._submit()
        return job

    # -------------------- exact evaluation --------------------

    def _run_exact(self, pubs: Iterable[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        """Deterministically evaluate all PUBs.

        Args:
            pubs (Iterable[SamplerPub]): Fully coerced PUBs.

        Returns:
            PrimitiveResult[SamplerPubResult]: Exact results for each PUB.
        """
        results = [self._run_pub_exact(pub) for pub in pubs]
        return PrimitiveResult(results)

    def _run_pub_exact(self, pub: SamplerPub) -> SamplerPubResult:
        """Compute per-register exact probability containers for a single PUB.

        Args:
            pub (SamplerPub): PUB containing circuit and parameters.

        Returns:
            SamplerPubResult: Exact probability results for this PUB.
        """
        unitary_circ, qargs, meas_info = _preprocess_circuit(pub.circuit)

        bound_circuits = pub.parameter_values.bind_all(unitary_circ)

        # For each bound config, compute exact joint probabilities over measured qubits.
        joint_probs_per_index = np.empty(bound_circuits.shape, dtype=object)
        for index, circ in np.ndenumerate(bound_circuits):
            if qargs:
                sv = Statevector.from_instruction(circ)
                joint = sv.probabilities_dict(qargs=qargs)
            else:
                joint = {"": 1.0}
            joint_probs_per_index[index] = joint

        # Build per-register ExactProbArray views (one per broadcast index)
        data_fields: dict[str, Any] = {}
        names: list[str] = []
        for item in meas_info:
            names.append(item.creg_name)

            arr = np.empty(bound_circuits.shape, dtype=object)
            for index, joint in np.ndenumerate(joint_probs_per_index):
                arr[index] = ExactProbArray(
                    joint_probs=joint,
                    mask=list(item.qreg_indices),
                    num_bits=item.num_bits,
                    shape=(),
                )

            # Wrap ND arrays so users can call .get_counts() / .get_probabilities()
            field_value: Any
            if arr.shape == ():
                field_value = arr.item()
            else:
                field_value = ExactProbNDArray(arr)

            data_fields[item.creg_name] = field_value

        # Package DataBin and return our result subclass.
        data_bin = DataBin(**data_fields, shape=bound_circuits.shape)
        return _ExactSamplerPubResult(
            data_bin,
            metadata={
                "shots": None,
                "exact": True,
                "names": names,
                "circuit_metadata": getattr(pub, "metadata", {}),
            },
        )


# -------------------- deterministic probability containers --------------------


class ExactProbArray:
    """
    Deterministic probability container (scalar, i.e. shape == ()).
    Methods:
      - get_probabilities(loc=None) -> dict[str, float]
      - get_counts(loc=None, shots=None) -> dict[str, int]  # only if distribution is dyadic
    Supports concatenation via concatenate_bits() so join_data() forms the exact joint.
    """

    __slots__ = ("_joint_probs", "_mask", "_num_bits", "_shape")

    def __init__(
        self,
        joint_probs: Mapping[str, float],  # over the full measured bitstring
        mask: list[int],  # LSB-based indices this register exposes
        num_bits: int,
        shape: tuple[int, ...] = (),
    ):
        """Exact probability container for a single classical register.

        Args:
            joint_probs (Mapping[str, float]): Full joint measured-bit distribution.
            mask (list[int]): LSB-ordered indices selecting bits exposed by this register.
            num_bits (int): Width of the classical register.
            shape (tuple[int, ...]): Broadcast shape (default ()).
        """
        self._joint_probs = dict(joint_probs)
        self._mask = list(mask)
        self._num_bits = int(num_bits)
        self._shape = tuple(shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the broadcast shape of this container."""
        return self._shape

    @property
    def num_bits(self) -> int:
        """Return the number of classical bits represented by this container."""
        return self._num_bits

    @property
    def num_shots(self):
        """Return None to indicate that this distribution is analytic, not sampled."""
        return None

    def _project_joint_to_mask(self, probs: Mapping[str, float]) -> dict[str, float]:
        """Project the joint distribution to this register's bit mask.

        Args:
            probs (Mapping[str, float]): Full joint distribution.

        Returns:
            dict[str, float]: Marginalized probability distribution.
        """
        out: dict[str, float] = {}
        for bitstr, p in probs.items():
            bits = list(bitstr)  # left
            sel = [bits[-1 - i] for i in reversed(self._mask)]  # LSB index 0 is rightmost char
            key = "".join(sel)
            out[key] = out.get(key, 0.0) + p
        return out

    def get_probabilities(self) -> dict[str, float]:
        """Return exact bitstring probabilities.

        Returns:
            dict[str, float]: Map from bitstring to exact probability.
        """
        return self._project_joint_to_mask(self._joint_probs)

    def get_counts(self, shots: int | None = None) -> dict[str, int]:
        """Return dyadic counts consistent with probabilities.

        Args:
            shots (int | None): Number of counts to generate. If ``None``, use dyadic size.

        Returns:
            dict[str, int]: Counts per bitstring.

        Raises:
            ValueError: If the distribution is not dyadic.
        """
        probs = self.get_probabilities()

        def dyadic_k(p: float, tol=1e-12, kmax=60) -> int | None:
            """Helper function"""
            if p in (0.0, 1.0):
                return 0
            for k in range(kmax + 1):
                m = round(p * (1 << k))
                if abs(p - m / float(1 << k)) <= tol:
                    return k
            return None

        ks = []
        for p in probs.values():
            k = dyadic_k(p)
            if k is None:
                raise ValueError(
                    "ExactProbArray.get_counts: distribution is not dyadic; "
                    "use get_probabilities() for exact values."
                )
            ks.append(k)
        k_common = max(ks) if ks else 0
        M = (1 << k_common) if shots is None else int(shots)  # pylint: disable=invalid-name

        counts: dict[str, int] = {k: int(round(v * M)) for k, v in probs.items()}
        total = sum(counts.values())
        if shots is None and counts and total != M:
            # Adjust the most likely entry to make totals consistent.
            key_star = max(probs, key=probs.get)
            counts[key_star] += M - total
        return counts

    @staticmethod
    def concatenate_bits(items: list["ExactProbArray"]) -> "ExactProbArray":
        """Concatenate multiple ``ExactProbArray`` instances.

        Args:
            items (list[ExactProbArray]): Containers to concatenate.

        Returns:
            ExactProbArray: Wider register combining all bits.

        Raises:
            ValueError: If the joint distributions are incompatible.
        """
        if not items:
            raise ValueError("No containers to concatenate.")
        joint = items[0]._joint_probs
        for it in items[1:]:
            if it._joint_probs is not joint and it._joint_probs != joint:
                raise ValueError("Cannot join different joint distributions.")
        mask: list[int] = []
        for it in items:
            mask.extend(it._mask)
        num_bits = sum(it._num_bits for it in items)
        return ExactProbArray(joint, mask=mask, num_bits=num_bits, shape=items[0]._shape)


class ExactProbNDArray:
    """
    ND wrapper around a numpy ndarray of ExactProbArray (dtype=object).
    Exposes SamplerV2-like methods on the whole array:
      - .get_counts(loc=None, shots=None)
      - .get_probabilities(loc=None)
    Supports indexing with numpy semantics: obj[idx].
    """

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        """N-dimensional wrapper for arrays of ``ExactProbArray``.

        Args:
            arr (np.ndarray): Object array of ``ExactProbArray`` elements.
        """
        self._arr = arr

    # --- array-like protocol ---
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying array."""
        return self._arr.shape

    def __getitem__(self, idx: int | tuple[int, ...] | None) -> Any:
        """Return probabilities element-wise or at a specific index.

        Args:
            loc (int | tuple[int, ...] | None): Optional index.

        Returns:
            dict[str, float] | np.ndarray: Probabilities for the selected element or array.
        """
        out = self._arr[idx]
        # Preserve behavior: if slicing returns an ndarray of ExactProbArray, wrap again.
        if isinstance(out, np.ndarray):
            return ExactProbNDArray(out)
        return out  # single ExactProbArray

    # Optional, used by some user code
    @property
    def num_shots(self):
        """Return None to indicate that all elements represent analytic distributions."""
        return None

    @property
    def num_bits(self) -> int:
        """Return the number of bits per element, inferred from a representative
        ExactProbArray."""
        # Uniform across elements
        _ = next(np.nditer(np.empty((1,), dtype=object), flags=[], op_flags=[]), None)
        try:
            # find a representative element
            rep = next(x for x in self._arr.flat if isinstance(x, ExactProbArray))
            return rep.num_bits
        except StopIteration:
            return 0

    # --- Sampler-style methods ---
    def get_probabilities(self, loc: int | tuple[int, ...] | None = None):
        """Return probabilities for a single location or an array of probability dicts for
        all entries."""
        if loc is not None:
            return self._arr[loc].get_probabilities()
        # Return element-wise probabilities as an ndarray[object] of dicts
        out = np.empty(self._arr.shape, dtype=object)
        for idx in np.ndindex(self._arr.shape):
            out[idx] = self._arr[idx].get_probabilities()  # type: ignore
        return out

    def get_counts(
        self, loc: int | tuple[int, ...] | None = None, shots: int | None = None
    ) -> dict[str, int] | np.ndarray:
        """Return counts element-wise or the union across positions.

        Args:
            loc (int | tuple[int, ...] | None): Optional index.
            shots (int | None): Number of shots for counts.

        Returns:
            dict[str, int] | np.ndarray: Counts for the selected element or union.
        """
        if loc is not None:
            return self._arr[loc].get_counts(shots=shots)

        # When location=None, follow BitArray semantics: union counts across all positions.
        # If you want per-position, index first (e.g., obj[i].get_counts()).
        total: dict[str, int] = {}
        for elem in self._arr.flat:
            # for exact non-dyadic distributions this raises; caller can use get_probabilities instead
            cnt = elem.get_counts(shots=shots)
            for k, v in cnt.items():
                total[k] = total.get(k, 0) + v
        return total


# --- helpers -------------------------------------------------


def _options_to_dict(opts) -> dict:
    """Convert an options object to a plain dict.

    Args:
        opts: Any options-like object.

    Returns:
        dict: Extracted key–value pairs.
    """
    if opts is None:
        return {}
    if is_dataclass(opts):
        return asdict(opts)  # type: ignore
    if hasattr(opts, "__dict__"):
        return {k: v for k, v in vars(opts).items() if not k.startswith("_")}
    # Fallback: probe attributes
    d = {}
    for k in dir(opts):
        if k.startswith("_"):
            continue
        v = getattr(opts, k)
        if callable(v):
            continue
        d[k] = v
    return d


class _OptionsNS(SimpleNamespace):
    """Mutable, dict-like options name space with an update(**kwargs) helper."""

    def update(self, **kwargs):
        """Update options in place.

        Args:
            **kwargs: Key–value pairs to update.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------- measurement mapping from StatevectorSampler --------------------


@dataclass
class _MeasureInfo:
    """Return a map from each final classical bit to the qubit index it measures, assuming
    only final measurements."""

    creg_name: str
    num_bits: int  # measured bit-width of this register
    qreg_indices: list[int]  # LSB-order indices into the joint measured-qubit list


def _final_measurement_mapping(circuit: QuantumCircuit) -> dict[tuple[ClassicalRegister, int], int]:
    """Map final classical bits to qubit indices.

    Args:
        circuit (QuantumCircuit): Circuit with final measurements.

    Returns:
        dict[(ClassicalRegister, int), int]: Mapping from classical bit to qubit index.
    """
    active_qubits = set(range(circuit.num_qubits))
    active_cbits = set(range(circuit.num_clbits))
    mapping: dict[tuple[ClassicalRegister, int], int] = {}
    for inst in circuit[::-1]:
        op = inst.operation.name
        if op == "measure":
            loc = circuit.find_bit(inst.clbits[0])
            c_idx = loc.index
            q_idx = circuit.find_bit(inst.qubits[0]).index
            if c_idx in active_cbits and q_idx in active_qubits:
                for creg in loc.registers:  # (ClassicalRegister, offset within that register)
                    mapping[creg] = q_idx
                active_cbits.remove(c_idx)
        elif op not in ("barrier", "delay"):
            for q in inst.qubits:
                q_i = circuit.find_bit(q).index
                active_qubits.discard(q_i)
        if not active_cbits or not active_qubits:
            break
    return mapping


def _preprocess_circuit(circuit: QuantumCircuit):
    """Preprocess a circuit to extract measurement mapping.

    Args:
        circuit (QuantumCircuit): Circuit with final measurements.

    Returns:
        tuple:
            QuantumCircuit: Circuit with final measurements removed.
            list[int]: Sorted measured qubit indices.
            list[_MeasureInfo]: Measurement metadata per classical register.
    """
    mapping = _final_measurement_mapping(circuit)
    qargs = sorted(set(mapping.values()))
    qargs_index = {q: i for i, q in enumerate(qargs)}
    unitary_circ = circuit.remove_final_measurements(inplace=False)

    # Keep classical-register bit order for masks.
    by_reg: dict[str, list[tuple[int, int]]] = {creg.name: [] for creg in circuit.cregs}
    for (creg, offset), q in mapping.items():
        by_reg[creg.name].append((offset, qargs_index[q]))  # (lsb_index_in_creg, joint_index)

    meas_info: list[_MeasureInfo] = []
    for name, pairs in by_reg.items():
        if not pairs:
            continue
        pairs.sort(key=lambda t: t[0])  # LSB-first
        mask = [joint for (_, joint) in pairs]  # mask in LSB order
        meas_info.append(_MeasureInfo(creg_name=name, num_bits=len(mask), qreg_indices=mask))

    return unitary_circ, qargs, meas_info


# ---------------------- PubResult subclass with safe join_data ----------------------


class _ExactSamplerPubResult(SamplerPubResult):
    """SamplerPubResult variant whose join_data() understands ExactProbArray and
    ExactProbNDArray containers."""

    def join_data(self, names: Iterable[str] | None = None):
        """Join named per-register probability containers.

        Args:
            names (Iterable[str] | None): Register names to join.

        Returns:
            ExactProbArray | ExactProbNDArray: Concatenated bit container.

        Raises:
            ValueError: If names are empty or missing.
        """
        if names is None:
            names = list(self.metadata.get("names", []))
        names = list(names)
        if not names:
            raise ValueError("names is empty")
        for n in names:
            if not hasattr(self.data, n):
                raise ValueError(f"name does not exist: {n}")

        shape = self.data.shape
        if shape == ():
            # Scalar: concatenate and return a single ExactProbArray
            items: list[ExactProbArray] = []
            for n in names:
                field = getattr(self.data, n)
                items.append(field)  # field is ExactProbArray
            return ExactProbArray.concatenate_bits(items)

        # ND case: build an ndarray of ExactProbArray and return a wrapper
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(shape):
            items: list[ExactProbArray] = []  # type: ignore
            for n in names:
                field = getattr(self.data, n)  # can be ExactProbNDArray
                field_elem = field[idx] if isinstance(field, ExactProbNDArray) else field[idx]
                items.append(field_elem)
            out[idx] = ExactProbArray.concatenate_bits(items)
        return ExactProbNDArray(out)
