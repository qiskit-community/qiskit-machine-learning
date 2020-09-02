# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
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
from test.ml import QiskitMLTestCase


class TestReadmeSample(QiskitMLTestCase):
    """Test sample code from readme"""

    def test_readme_sample(self):
        """ readme sample test """
        # pylint: disable=import-outside-toplevel,redefined-builtin

        def print(*args):
            """ overloads print to log values """
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        from qiskit import BasicAer
        from qiskit.aqua import QuantumInstance, aqua_globals
        from qiskit.aqua.algorithms import VQC
        from qiskit.aqua.components.optimizers import COBYLA
        from qiskit.aqua.components.feature_maps import RawFeatureVector
        from qiskit.ml.datasets import wine
        from qiskit.circuit.library import TwoLocal

        seed = 1376
        aqua_globals.random_seed = seed

        # Use Wine data set for training and test data
        feature_dim = 4  # dimension of each data point
        _, training_input, test_input, _ = wine(training_size=12,
                                                test_size=4,
                                                n=feature_dim)

        feature_map = RawFeatureVector(feature_dimension=feature_dim)
        vqc = VQC(COBYLA(maxiter=100),
                  feature_map,
                  TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
                  training_input,
                  test_input)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024, seed_simulator=seed, seed_transpiler=seed))

        print('Testing accuracy: {:0.2f}'.format(result['testing_accuracy']))

        # ----------------------------------------------------------------------

        self.assertGreater(result['testing_accuracy'], 0.8)


if __name__ == '__main__':
    unittest.main()
