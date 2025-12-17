# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test primitives that check what kind of operations are in the circuits they execute."""

from qiskit_machine_learning.primitives import QMLEstimator, QMLSampler


class LoggingEstimator(QMLEstimator):
    """An estimator checking what operations were in the circuits it executed."""

    def __init__(self, operations_callback=None):
        super().__init__(estimator=None)
        self.operations_callback = operations_callback

    def run(self, pubs, **run_options):
        if self.operations_callback is not None:
            ops = [pub[0].count_ops() for pub in pubs]
            self.operations_callback(ops)
        return super().run(pubs, **run_options)


class LoggingSampler(QMLSampler):
    """A sampler checking what operations were in the circuits it executed."""

    def __init__(self, operations_callback):
        super().__init__()
        self.operations_callback = operations_callback

    def run(self, pubs, **run_options):
        ops = [pub[0].count_ops() for pub in pubs]
        self.operations_callback(ops)
        return super().run(pubs, **run_options)
