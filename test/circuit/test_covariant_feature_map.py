# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ``CovariantFeatureMap`` circuit."""

from test import QiskitMachineLearningTestCase
from qiskit_machine_learning.circuit.library import CovariantFeatureMap


class TestCovariantFeatureMap(QiskitMachineLearningTestCase):
    """Test the ``CovariantFeatureMap`` circuit."""

    def test_construction_with_training_parameters(self):
        """Test construction of ``CovariantFeatureMap``."""

        circuit = CovariantFeatureMap(
            feature_dimension=6,
            entanglement=[[0, 2], [2, 1]],
            include_training_parameters=True,
        )

        with self.subTest("check circuit built"):
            self.assertEqual(circuit.num_qubits, 3)
            self.assertEqual(len(circuit.input_parameters), 6)
            self.assertEqual(len(circuit.training_parameters), 3)
            self.assertEqual(circuit.num_parameters, 9)

    def test_construction_without_training_parameters(self):
        """Test construction of ``CovariantFeatureMap``."""

        circuit = CovariantFeatureMap(
            feature_dimension=6,
            entanglement=[[0, 2], [2, 1]],
            include_training_parameters=False,
        )

        with self.subTest("check circuit built"):
            self.assertEqual(circuit.num_qubits, 3)
            self.assertEqual(len(circuit.input_parameters), 6)
            self.assertEqual(len(circuit.training_parameters), 0)
            self.assertEqual(circuit.num_parameters, 6)

    def test_construction_fails(self):
        """Test invalid construction."""

        with self.assertRaisesRegex(ValueError, "even number"):
            CovariantFeatureMap(feature_dimension=3)

        with self.assertRaises(TypeError):
            CovariantFeatureMap()

        with self.assertRaises(IndexError):
            CovariantFeatureMap(
                feature_dimension=4,
                entanglement=[[0, 2]],
            )
