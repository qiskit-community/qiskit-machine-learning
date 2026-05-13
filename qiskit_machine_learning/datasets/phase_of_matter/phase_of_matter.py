# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Public API for the Phase of Matter dataset generator."""

from __future__ import annotations

import math

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from . import _annni, _cluster, _haldane, _heisenberg
from ._base import get_ground_state_exact, get_ground_state_vqe

# ---------------------------------------------------------------------------
# Registry — maps model name to its module
# ---------------------------------------------------------------------------

_MODELS = {
    "heisenberg": _heisenberg,
    "haldane": _haldane,
    "annni": _annni,
    "cluster": _cluster,
}

_BUILDERS = {
    "heisenberg": lambda n, p: _heisenberg.build_hamiltonian(n, p["j1"], p["j2"]),
    "haldane": lambda n, p: _haldane.build_hamiltonian(n, p["h1"], p["h2"]),
    "annni": lambda n, p: _annni.build_hamiltonian(n, p["kappa"], p["h"]),
    "cluster": lambda n, p: _cluster.build_hamiltonian(n, p["j1"], p["j2"]),
}


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def phase_of_matter_data(
    training_size: int,
    test_size: int,
    n: int,
    *,
    model: str = "heisenberg",
    one_hot: bool = True,
    include_sample_total: bool = False,
    class_labels: list | None = None,
    formatting: str = "ndarray",
    seed: int | None = None,
    backend=None,
) -> (
    tuple[np.ndarray | list[Statevector], np.ndarray, np.ndarray | list[Statevector], np.ndarray]
    | tuple[
        np.ndarray | list[Statevector],
        np.ndarray,
        np.ndarray | list[Statevector],
        np.ndarray,
        np.ndarray,
    ]
):
    r"""Generate a quantum Phase of Matter classification dataset.

    For each sample, coupling parameters are drawn uniformly from the interior
    of a known phase region, the corresponding Hamiltonian is built as a
    :class:`~qiskit.quantum_info.SparsePauliOp`, and its ground state is
    computed via sparse exact diagonalization.  The ground-state vector forms
    the feature, and the phase name forms the label.

    Four spin-chain Hamiltonians are supported (see the reference for the
    exact definitions and phase diagrams):

    * ``"heisenberg"`` — Bond-alternating XXX Heisenberg model (eq. 6).
      Phases: *trivial*, *topological*.
    * ``"haldane"`` — Haldane chain (eq. 7).
      Phases: *antiferromagnetic*, *paramagnetic*, *spt*.
    * ``"annni"`` — Axial Next-Nearest-Neighbor Ising model (eq. 8).
      Phases: *ferromagnetic*, *paramagnetic*, *floating*, *antiphase*.
    * ``"cluster"`` — Cluster Hamiltonian with periodic boundary (eq. 9).
      Phases: *haldane*, *ferromagnetic*, *antiferromagnetic*, *trivial*.

    Args:
        training_size: Total number of training samples (balanced across
            classes).
        test_size: Total number of test samples (balanced across classes).
        n: Number of lattice sites (qubits).  Must be ≥ 4.  The feature
            dimension is :math:`2^n`; practical limit for exact
            diagonalization is ``n ≤ 16``.
        model: Hamiltonian to use.  One of ``"heisenberg"``, ``"haldane"``,
            ``"annni"``, ``"cluster"``.
        one_hot: If ``True`` (default), labels are one-hot encoded numpy
            arrays.  If ``False``, string phase names are returned.
        include_sample_total: If ``True``, a fifth element is appended to the
            return tuple with the number of ground states computed per class.
        class_labels: Optional list of custom label names that replace the
            model's default phase names.  Length must equal the number of
            phases for the chosen model.
        formatting: ``"ndarray"`` (default) returns features as a complex
            numpy array of shape ``(num_samples, 2**n)``.
            ``"statevector"`` returns a list of
            :class:`~qiskit.quantum_info.Statevector` objects.
        seed: Integer seed for the parameter-sampling RNG, enabling
            reproducible datasets.
        backend: When ``None`` (default), exact diagonalization via
            ``scipy.sparse.linalg.eigsh`` is used — the recommended path for
            reliable phase labels.  When a Qiskit backend is provided, a
            VQE-based approximation is used instead.

            .. warning::

                The VQE pathway is for hardware-experiment workflows only.
                VQE approximations near phase boundaries may produce
                incorrect labels.  Use ``backend=None`` for dataset
                generation.

    Returns:
        A tuple ``(training_features, training_labels, test_features,
        test_labels)`` where:

        * ``training_features`` / ``test_features`` — shape
          ``(n_samples, 2**n)`` complex ndarray, or list of
          :class:`~qiskit.quantum_info.Statevector` when
          ``formatting="statevector"``.
        * ``training_labels`` / ``test_labels`` — shape
          ``(n_samples, n_classes)`` one-hot ndarray when ``one_hot=True``,
          or list of strings when ``one_hot=False``.

        If ``include_sample_total=True``, a fifth element — a numpy array of
        shape ``(n_classes,)`` containing the number of ground states
        computed per class — is appended.

    Raises:
        ValueError: If *model* is not one of the supported strings.
        ValueError: If *formatting* is not ``"ndarray"`` or
            ``"statevector"``.
        ValueError: If ``n < 4``.
        ValueError: If *class_labels* is provided but has the wrong length.

    References:
        [1] Bermejo et al., "Quantum Convolutional Neural Networks are
        (Effectively) Classically Simulable", arXiv:2408.12739 (2024).

    Examples:

        >>> x_tr, y_tr, x_te, y_te = phase_of_matter_data(
        ...     10, 5, 4, model="heisenberg", seed=0
        ... )
        >>> x_tr.shape
        (10, 16)
        >>> y_tr.shape
        (10, 2)
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {sorted(_MODELS.keys())}.")
    if formatting not in ("ndarray", "statevector"):
        raise ValueError(f"Unknown formatting '{formatting}'. Choose 'ndarray' or 'statevector'.")
    if n < 4:
        raise ValueError(f"n must be at least 4, got {n}.")

    module = _MODELS[model]
    default_labels: list[str] = module.PHASE_LABELS
    n_classes = len(default_labels)

    if class_labels is not None:
        if len(class_labels) != n_classes:
            raise ValueError(
                f"class_labels has {len(class_labels)} entries but model '{model}' "
                f"has {n_classes} phases."
            )
        label_names = list(class_labels)
    else:
        label_names = list(default_labels)

    rng = np.random.default_rng(seed)

    # ceil ensures every class gets at least the requested count even when
    # training_size / test_size are not divisible by n_classes.
    n_per_class_train = math.ceil(training_size / n_classes)
    n_per_class_test = math.ceil(test_size / n_classes)
    n_per_class = n_per_class_train + n_per_class_test

    # Samplers return blocks of n_per_class per class, class order preserved.
    raw_samples = module.sample_parameters(n_per_class, rng)

    build_fn = _BUILDERS[model]
    gs_fn = (
        (lambda h: get_ground_state_vqe(h, backend))
        if backend is not None
        else get_ground_state_exact
    )

    # Compute ground states — preserve class-block order for the split below.
    all_states: list[np.ndarray] = []
    all_labels: list[str] = []
    for params, phase in raw_samples:
        H: SparsePauliOp = build_fn(n, params)
        gs = gs_fn(H)
        if isinstance(gs, Statevector):
            gs = gs.data
        all_states.append(gs)
        idx = default_labels.index(phase)
        all_labels.append(label_names[idx])

    # Split per class into train / test.
    train_states: list[np.ndarray] = []
    train_labels_raw: list[str] = []
    test_states: list[np.ndarray] = []
    test_labels_raw: list[str] = []
    sample_totals = np.zeros(n_classes, dtype=int)

    for cls_idx in range(n_classes):
        start = cls_idx * n_per_class
        cls_states = all_states[start : start + n_per_class]
        cls_labels = all_labels[start : start + n_per_class]
        train_states.extend(cls_states[:n_per_class_train])
        train_labels_raw.extend(cls_labels[:n_per_class_train])
        test_states.extend(cls_states[n_per_class_train:])
        test_labels_raw.extend(cls_labels[n_per_class_train:])
        sample_totals[cls_idx] = n_per_class

    # Trim to exact requested sizes (ceil may over-allocate by up to n_classes-1).
    train_states = train_states[:training_size]
    train_labels_raw = train_labels_raw[:training_size]
    test_states = test_states[:test_size]
    test_labels_raw = test_labels_raw[:test_size]

    # Shuffle train and test independently to interleave classes.
    tr_idx = np.arange(len(train_states))
    rng.shuffle(tr_idx)
    te_idx = np.arange(len(test_states))
    rng.shuffle(te_idx)
    train_states = [train_states[i] for i in tr_idx]
    train_labels_raw = [train_labels_raw[i] for i in tr_idx]
    test_states = [test_states[i] for i in te_idx]
    test_labels_raw = [test_labels_raw[i] for i in te_idx]

    # Format features.
    if formatting == "ndarray":
        x_train: np.ndarray | list[Statevector] = np.array(train_states)
        x_test: np.ndarray | list[Statevector] = np.array(test_states)
    else:
        x_train = [Statevector(s) for s in train_states]
        x_test = [Statevector(s) for s in test_states]

    # Format labels.
    label_to_idx = {lbl: i for i, lbl in enumerate(label_names)}

    def _make_labels(raw: list[str]) -> np.ndarray:
        if one_hot:
            mat = np.zeros((len(raw), n_classes), dtype=float)
            for row, lbl in enumerate(raw):
                mat[row, label_to_idx[lbl]] = 1.0
            return mat
        return np.array(raw)

    y_train = _make_labels(train_labels_raw)
    y_test = _make_labels(test_labels_raw)

    if include_sample_total:
        return x_train, y_train, x_test, y_test, sample_totals
    return x_train, y_train, x_test, y_test
