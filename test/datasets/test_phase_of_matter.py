# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2024, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the Phase of Matter dataset generator.

Follows qiskit-machine-learning test conventions:
  - QiskitMachineLearningTestCase base class
  - parameterized tests via the ddt library
  - np.testing.assert_* for array assertions
"""

from __future__ import annotations

import unittest
from test import QiskitMachineLearningTestCase
from typing import Callable

import numpy as np
from ddt import ddt, idata, unpack
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit_machine_learning.datasets import phase_of_matter_data
from qiskit_machine_learning.datasets.phase_of_matter._annni import (
    build_hamiltonian as build_annni,
)
from qiskit_machine_learning.datasets.phase_of_matter._base import get_ground_state_exact
from qiskit_machine_learning.datasets.phase_of_matter._cluster import (
    build_hamiltonian as build_cluster,
)
from qiskit_machine_learning.datasets.phase_of_matter._haldane import (
    build_hamiltonian as build_haldane,
)
from qiskit_machine_learning.datasets.phase_of_matter._heisenberg import (
    build_hamiltonian as build_heisenberg,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _is_hermitian(op: SparsePauliOp, tol: float = 1e-10) -> bool:
    """Return True if op is Hermitian within the given tolerance."""
    mat = op.to_matrix()
    return np.allclose(mat, mat.conj().T, atol=tol)


# ---------------------------------------------------------------------------
# TestHamiltonianBuilders
# ---------------------------------------------------------------------------


@ddt
class TestHamiltonianBuilders(QiskitMachineLearningTestCase):
    """Verify that each Hamiltonian builder returns a valid Hermitian operator."""

    @idata([(4,), (6,)])
    @unpack
    def test_heisenberg_hermitian(self, n):
        """Heisenberg Hamiltonian must be Hermitian."""
        ham = build_heisenberg(n, j1=1.0, j2=0.5)
        self.assertTrue(_is_hermitian(ham), f"Heisenberg n={n} is not Hermitian")

    @idata([(4,), (6,)])
    @unpack
    def test_haldane_hermitian(self, n):
        """Haldane Hamiltonian must be Hermitian."""
        ham = build_haldane(n, h1=0.5, h2=0.3)
        self.assertTrue(_is_hermitian(ham), f"Haldane n={n} is not Hermitian")

    @idata([(4,), (6,)])
    @unpack
    def test_annni_hermitian(self, n):
        """ANNNI Hamiltonian must be Hermitian."""
        ham = build_annni(n, kappa=0.3, h=0.5)
        self.assertTrue(_is_hermitian(ham), f"ANNNI n={n} is not Hermitian")

    @idata([(4,), (6,)])
    @unpack
    def test_cluster_hermitian(self, n):
        """Cluster Hamiltonian must be Hermitian."""
        ham = build_cluster(n, j1=1.0, j2=-1.0)
        self.assertTrue(_is_hermitian(ham), f"Cluster n={n} is not Hermitian")

    def test_cluster_periodic_boundary(self):
        """Cluster Hamiltonian must have more terms than diagonal Z terms alone."""
        n = 4
        ham = build_cluster(n, j1=1.0, j2=1.0)
        # n Z terms + n XX two-body terms + n Z-X-Z three-body terms = 3n unique terms minimum
        self.assertGreater(len(ham), n)

    def test_matrix_dimension(self):
        """All models should produce a 2^n x 2^n matrix for n=4."""
        n = 4
        dim = 2**n
        hamiltonians = [
            build_heisenberg(n, 1.0, 0.5),
            build_haldane(n, 0.5, 0.3),
            build_annni(n, 0.3, 0.5),
            build_cluster(n, 1.0, -1.0),
        ]
        for ham in hamiltonians:
            mat = ham.to_matrix()
            self.assertEqual(mat.shape, (dim, dim))


# ---------------------------------------------------------------------------
# TestGroundState
# ---------------------------------------------------------------------------


@ddt
class TestGroundState(QiskitMachineLearningTestCase):
    """Verify exact-diagonalization ground-state properties."""

    def _fixed_hamiltonian(self, model: str, n: int) -> SparsePauliOp:
        """Return a Hamiltonian at fixed safe parameters for the given model."""
        params: dict[str, tuple] = {
            "heisenberg": (n, 1.0, 0.5),
            "haldane": (n, 0.5, 0.3),
            "annni": (n, 0.3, 0.5),
            "cluster": (n, 1.0, -1.0),
        }
        builders: dict[str, Callable[..., SparsePauliOp]] = {
            "heisenberg": build_heisenberg,
            "haldane": build_haldane,
            "annni": build_annni,
            "cluster": build_cluster,
        }
        return builders[model](*params[model])

    @idata([("heisenberg",), ("haldane",), ("annni",), ("cluster",)])
    @unpack
    def test_normalization(self, model):
        """Ground state must be normalized to unit norm."""
        ham = self._fixed_hamiltonian(model, n=4)
        gs = get_ground_state_exact(ham)
        self.assertAlmostEqual(
            np.linalg.norm(gs), 1.0, places=8, msg=f"{model} ground state is not normalized"
        )

    @idata([("heisenberg",), ("haldane",), ("annni",), ("cluster",)])
    @unpack
    def test_is_eigenstate(self, model):
        """H|psi> must equal E|psi> up to numerical noise."""
        ham = self._fixed_hamiltonian(model, n=4)
        gs = get_ground_state_exact(ham)
        mat = ham.to_matrix()
        h_psi = mat @ gs
        energy = np.dot(gs.conj(), h_psi).real
        residual = np.linalg.norm(h_psi - energy * gs)
        self.assertLess(residual, 1e-8, msg=f"{model} eigenstate residual {residual:.2e}")

    def test_lowest_eigenvalue(self):
        """Energy from eigsh must match the minimum eigenvalue from dense diagonalization."""
        ham = build_heisenberg(4, j1=1.0, j2=2.0)
        gs = get_ground_state_exact(ham)
        mat = ham.to_matrix()
        e_eigsh = (gs.conj() @ mat @ gs).real
        e_min = np.linalg.eigvalsh(mat).min()
        self.assertAlmostEqual(e_eigsh, e_min, places=8)


# ---------------------------------------------------------------------------
# TestPhaseLabels
# ---------------------------------------------------------------------------


@ddt
class TestPhaseLabels(QiskitMachineLearningTestCase):
    """Verify that phase-sampling regions produce correct labels."""

    @idata(
        [
            (0.2, "trivial"),
            (2.5, "topological"),
        ]
    )
    @unpack
    def test_heisenberg_phase_region(
        self, j2_ratio, expected_label
    ):  # pylint: disable=unused-argument
        """Heisenberg labels sampled far from boundary must include expected phase."""
        # j2_ratio is the parameter value used in the docstring example but the
        # sampler draws from fixed interior regions; we verify that both phases
        # appear across a dataset generated from the full interior.
        _, y, _, _ = phase_of_matter_data(20, 4, 4, model="heisenberg", one_hot=False, seed=0)
        self.assertIn(expected_label, set(y), msg=f"Label '{expected_label}' missing from dataset")

    @idata(
        [
            ("ferromagnetic",),
            ("paramagnetic",),
            ("floating",),
            ("antiphase",),
        ]
    )
    @unpack
    def test_annni_all_phases_present(self, phase):
        """All four ANNNI phases must appear in a sufficiently large dataset."""
        _, y, _, _ = phase_of_matter_data(40, 8, 4, model="annni", one_hot=False, seed=42)
        self.assertIn(phase, set(y), msg=f"ANNNI phase '{phase}' missing from dataset")

    @idata(
        [
            ("haldane",),
            ("ferromagnetic",),
            ("antiferromagnetic",),
            ("trivial",),
        ]
    )
    @unpack
    def test_cluster_all_phases_present(self, phase):
        """All four Cluster phases must appear in a sufficiently large dataset."""
        _, y, _, _ = phase_of_matter_data(40, 8, 4, model="cluster", one_hot=False, seed=42)
        self.assertIn(phase, set(y), msg=f"Cluster phase '{phase}' missing from dataset")

    @idata(
        [
            ("antiferromagnetic",),
            ("paramagnetic",),
            ("spt",),
        ]
    )
    @unpack
    def test_haldane_all_phases_present(self, phase):
        """All three Haldane phases must appear in a sufficiently large dataset."""
        _, y, _, _ = phase_of_matter_data(30, 6, 4, model="haldane", one_hot=False, seed=42)
        self.assertIn(phase, set(y), msg=f"Haldane phase '{phase}' missing from dataset")


# ---------------------------------------------------------------------------
# TestPublicAPI
# ---------------------------------------------------------------------------


@ddt
class TestPublicAPI(QiskitMachineLearningTestCase):
    """Verify the shape and type contracts of phase_of_matter_data."""

    @idata(
        [
            ("heisenberg", 2),
            ("haldane", 3),
            ("annni", 4),
            ("cluster", 4),
        ]
    )
    @unpack
    def test_return_shapes_ndarray(self, model, n_classes):
        """Feature and label arrays must have the correct shapes."""
        x_tr, y_tr, x_te, y_te = phase_of_matter_data(8, 4, 4, model=model, one_hot=True, seed=0)
        np.testing.assert_array_equal(x_tr.shape, (8, 16))
        np.testing.assert_array_equal(y_tr.shape, (8, n_classes))
        np.testing.assert_array_equal(x_te.shape, (4, 16))
        np.testing.assert_array_equal(y_te.shape, (4, n_classes))

    @idata([("heisenberg",), ("annni",)])
    @unpack
    def test_return_shapes_statevector(self, model):
        """Statevector formatting must return normalized Statevector objects."""
        x_tr, _, x_te, _ = phase_of_matter_data(
            4, 2, 4, model=model, formatting="statevector", seed=0
        )
        self.assertEqual(len(x_tr), 4)
        self.assertEqual(len(x_te), 2)
        self.assertIsInstance(x_tr[0], Statevector)
        self.assertAlmostEqual(np.linalg.norm(x_tr[0].data), 1.0, places=6)

    def test_one_hot_true_sums_to_one(self):
        """One-hot rows must each sum to exactly 1."""
        _, y_tr, _, _ = phase_of_matter_data(8, 4, 4, model="heisenberg", one_hot=True, seed=0)
        np.testing.assert_array_equal(y_tr.sum(axis=1), np.ones(8))

    def test_one_hot_false_returns_strings(self):
        """String labels must be a subset of the model's phase names."""
        _, y_tr, _, _ = phase_of_matter_data(8, 4, 4, model="heisenberg", one_hot=False, seed=0)
        self.assertTrue(all(isinstance(lbl, str) for lbl in y_tr))
        self.assertTrue(set(y_tr).issubset({"trivial", "topological"}))

    def test_include_sample_total_false(self):
        """Default return must be a 4-tuple."""
        result = phase_of_matter_data(4, 2, 4, model="heisenberg", seed=0)
        self.assertEqual(len(result), 4)

    def test_include_sample_total_true(self):
        """include_sample_total=True must append a per-class count array."""
        result = phase_of_matter_data(
            4, 2, 4, model="heisenberg", include_sample_total=True, seed=0
        )
        self.assertEqual(len(result), 5)
        totals = result[4]
        self.assertEqual(totals.shape, (2,))  # 2 classes for heisenberg
        self.assertTrue(np.all(totals > 0))

    def test_custom_class_labels(self):
        """Custom label names must replace the model defaults in string output."""
        _, y_tr, _, _ = phase_of_matter_data(
            8, 4, 4, model="heisenberg", one_hot=False, class_labels=["phase_A", "phase_B"], seed=0
        )
        self.assertTrue(set(y_tr).issubset({"phase_A", "phase_B"}))

    def test_custom_class_labels_one_hot(self):
        """Custom labels must not affect one-hot shape or values."""
        _, y1, _, _ = phase_of_matter_data(8, 4, 4, model="heisenberg", one_hot=True, seed=0)
        _, y2, _, _ = phase_of_matter_data(
            8, 4, 4, model="heisenberg", one_hot=True, class_labels=["A", "B"], seed=0
        )
        np.testing.assert_array_equal(y1, y2)

    def test_feature_normalization(self):
        """All returned ground states must be normalized."""
        x_tr, _, x_te, _ = phase_of_matter_data(8, 4, 4, model="annni", seed=1)
        for states in (x_tr, x_te):
            norms = np.linalg.norm(states, axis=1)
            np.testing.assert_allclose(
                norms, 1.0, atol=1e-8, err_msg="Ground states are not normalized"
            )

    def test_seed_reproducibility(self):
        """Same seed must produce numerically identical outputs.

        Features are complex floating-point arrays; we use allclose with a
        tight tolerance (1e-10) to allow for sub-machine-precision noise in
        the ARPACK eigensolver while still catching meaningful differences.
        """
        kwargs = dict(model="heisenberg", seed=99)
        x1, y1, xt1, yt1 = phase_of_matter_data(6, 3, 4, **kwargs)
        x2, y2, xt2, yt2 = phase_of_matter_data(6, 3, 4, **kwargs)
        np.testing.assert_allclose(
            x1, x2, atol=1e-10, err_msg="train features differ across equal seeds"
        )
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_allclose(
            xt1, xt2, atol=1e-10, err_msg="test features differ across equal seeds"
        )
        np.testing.assert_array_equal(yt1, yt2)

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different data."""
        x1, _, _, _ = phase_of_matter_data(8, 4, 4, model="heisenberg", seed=1)
        x2, _, _, _ = phase_of_matter_data(8, 4, 4, model="heisenberg", seed=2)
        self.assertFalse(np.allclose(x1, x2))

    def test_train_test_sizes_respected(self):
        """Exact training_size / test_size must be honored."""
        for tr, te in [(10, 3), (7, 7), (1, 1)]:
            x_tr, _, x_te, _ = phase_of_matter_data(tr, te, 4, model="heisenberg", seed=0)
            self.assertEqual(len(x_tr), tr, f"train size mismatch (requested {tr})")
            self.assertEqual(len(x_te), te, f"test size mismatch (requested {te})")

    # -----------------------------------------------------------------------
    # Error cases
    # -----------------------------------------------------------------------

    def test_invalid_model_raises(self):
        """An unknown model name must raise ValueError."""
        with self.assertRaises(ValueError):
            phase_of_matter_data(4, 2, 4, model="invalid")

    def test_invalid_formatting_raises(self):
        """An unknown formatting string must raise ValueError."""
        with self.assertRaises(ValueError):
            phase_of_matter_data(4, 2, 4, model="heisenberg", formatting="bad")

    def test_n_too_small_raises(self):
        """n < 4 must raise ValueError."""
        with self.assertRaises(ValueError):
            phase_of_matter_data(4, 2, 3, model="heisenberg")

    def test_wrong_class_labels_length_raises(self):
        """class_labels with wrong length must raise ValueError."""
        with self.assertRaises(ValueError):
            phase_of_matter_data(4, 2, 4, model="heisenberg", class_labels=["only_one"])


# ---------------------------------------------------------------------------
# Integration -- import paths
# ---------------------------------------------------------------------------


class TestImportPaths(QiskitMachineLearningTestCase):
    """Verify the package can be imported and is correctly wired up."""

    def test_importable(self):
        """phase_of_matter_data must be accessible from the datasets module."""
        import qiskit_machine_learning.datasets as ds  # pylint: disable=import-outside-toplevel

        self.assertIsNotNone(ds.phase_of_matter_data)

    def test_in_all(self):
        """phase_of_matter_data must be listed in datasets.__all__."""
        import qiskit_machine_learning.datasets as ds  # pylint: disable=import-outside-toplevel

        self.assertIn("phase_of_matter_data", ds.__all__)

    def test_hamiltonian_modules_importable(self):
        """All Hamiltonian sub-modules must expose the required attributes."""
        from qiskit_machine_learning.datasets.phase_of_matter import (  # pylint: disable=import-outside-toplevel
            _annni,
            _cluster,
            _haldane,
            _heisenberg,
        )

        for mod in (_heisenberg, _haldane, _annni, _cluster):
            self.assertTrue(hasattr(mod, "build_hamiltonian"))
            self.assertTrue(hasattr(mod, "sample_parameters"))
            self.assertTrue(hasattr(mod, "PHASE_LABELS"))


if __name__ == "__main__":
    unittest.main()
