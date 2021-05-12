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

"""
Code inside the test is the machine learning sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest

from test import QiskitMachineLearningTestCase


class TestReadmeSample(QiskitMachineLearningTestCase):
    """Test sample code from readme"""

    def test_readme_sample(self):
        """readme sample test"""
        # pylint: disable=import-outside-toplevel,redefined-builtin
        def print(*args):
            """overloads print to log values"""
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        from qiskit import BasicAer
        from qiskit.utils import QuantumInstance, algorithm_globals
        from qiskit.algorithms.optimizers import COBYLA
        from qiskit.circuit.library import TwoLocal
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_machine_learning.datasets import wine
        from qiskit_machine_learning.circuit.library import RawFeatureVector

        seed = 1376
        algorithm_globals.random_seed = seed

        # Use Wine data set for training and test data
        feature_dim = 4  # dimension of each data point
        training_size = 12
        test_size = 4

        # training features, training labels, test features, test labels as np.array,
        # one hot encoding for labels
        training_features, training_labels, test_features, test_labels = wine(
            training_size=training_size, test_size=test_size, n=feature_dim
        )

        feature_map = RawFeatureVector(feature_dimension=feature_dim)
        ansatz = TwoLocal(feature_map.num_qubits, ["ry", "rz"], "cz", reps=3)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=COBYLA(maxiter=100),
            quantum_instance=QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                shots=1024,
                seed_simulator=seed,
                seed_transpiler=seed,
            ),
        )
        vqc.fit(training_features, training_labels)

        score = vqc.score(test_features, test_labels)
        print("Testing accuracy: {:0.2f}".format(score))

        # ----------------------------------------------------------------------

        self.assertGreater(score, 0.7)


if __name__ == "__main__":
    unittest.main()
