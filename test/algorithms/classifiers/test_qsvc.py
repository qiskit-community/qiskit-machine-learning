# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QSVC """
import tempfile
import unittest
from pathlib import Path

from test import QiskitMachineLearningTestCase

import numpy as np

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, SerializableModelMixin
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.exceptions import (
    QiskitMachineLearningWarning,
)


class TestQSVC(QiskitMachineLearningTestCase):
    """Test QSVC Algorithm"""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.feature_map = zz_feature_map(feature_dimension=2, reps=2)

        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        self.label_train = np.asarray([0, 0, 1, 1])

        self.sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
        self.label_test = np.asarray([0, 1])

    def test_qsvc(self):
        """Test QSVC"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_change_kernel(self):
        """Test QSVC with FidelityQuantumKernel later"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        qsvc = QSVC()
        qsvc.quantum_kernel = qkernel
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_qsvc_parameters(self):
        """Test QSVC with extra constructor parameters"""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        qsvc = QSVC(quantum_kernel=qkernel, tol=1e-4, C=0.5)
        qsvc.fit(self.sample_train, self.label_train)
        score = qsvc.score(self.sample_test, self.label_test)

        self.assertEqual(score, 0.5)

    def test_qsvc_to_string(self):
        """Test QSVC print works when no *args passed in"""
        qsvc = QSVC()
        _ = str(qsvc)

    def test_with_kernel_parameter(self):
        """Test QSVC with the `kernel` argument."""
        with self.assertWarns(QiskitMachineLearningWarning):
            QSVC(kernel=1)

    def test_save_load_default_filename(self):
        """Saving and loading with default 'qsvc.pkl' should preserve predictions."""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        # temporary working directory
        tmpdir = Path(tempfile.mkdtemp())

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)
        orig_preds = qsvc.predict(self.sample_test)

        # save into subdirectory (should be auto-created)
        out_dir = tmpdir / "subfolder"
        qsvc.save(str(out_dir))

        # verify file exists
        pkl = out_dir / "qsvc.pkl"
        self.assertTrue(pkl.is_file(), f"{pkl} was not created")

        # load and compare
        loaded = QSVC.load(str(out_dir))
        load_preds = loaded.predict(self.sample_test)
        np.testing.assert_array_equal(orig_preds, load_preds)

    def test_save_load_custom_filename(self):
        """Saving and loading with a custom filename works identically."""
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        # temporary working directory
        tmpdir = Path(tempfile.mkdtemp())

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)
        orig_preds = qsvc.predict(self.sample_test)

        # use a custom filename
        fname = "my_model.bin"
        out_dir = tmpdir / "custom"
        qsvc.save(str(out_dir), filename=fname)

        full_path = out_dir / fname
        self.assertTrue(full_path.is_file())

        loaded = QSVC.load(str(out_dir), filename=fname)
        np.testing.assert_array_equal(orig_preds, loaded.predict(self.sample_test))

    def test_load_missing_file_raises(self):
        """Attempting to load from a missing file should raise FileNotFoundError."""
        tmpdir = Path(tempfile.mkdtemp())
        missing_dir = tmpdir / "nope"
        with self.assertRaises(FileNotFoundError):
            QSVC.load(str(missing_dir))

        # also if filename changed
        valid_dir = tmpdir / "empty"
        valid_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            QSVC.load(str(valid_dir), filename="does_not_exist.pkl")

    def test_load_wrong_type_raises(self):
        """Loading via SerializableModelMixin.load on a non-QSVC class should TypeError."""
        # first save a valid QSVC
        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)

        # temporary working directory
        tmpdir = Path(tempfile.mkdtemp())

        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)
        qsvc.save(str(tmpdir))

        # define a dummy class using the mixin
        class FakeModel(SerializableModelMixin):
            """Class that pretends to be QSVC"""

            pass

        # FakeModel.load should raise TypeError when unpickling a QSVC
        with self.assertRaises(TypeError):
            FakeModel.load(str(tmpdir / "qsvc.pkl"))

    def test_io_error_on_save_bad_folder(self):
        """If the folder cannot be created (e.g. file in its place), save should IOError."""
        # temporary working directory
        tmpdir = Path(tempfile.mkdtemp())
        # create a file where a folder should be
        bad = tmpdir / "not_a_folder"
        bad.write_text("I'm a file, not a directory")

        qkernel = FidelityQuantumKernel(feature_map=self.feature_map)
        qsvc = QSVC(quantum_kernel=qkernel)
        qsvc.fit(self.sample_train, self.label_train)

        with self.assertRaises(IOError):
            qsvc.save(str(bad / "sub"), filename="model.pkl")


if __name__ == "__main__":
    unittest.main()
