# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test primitives that check what kind of operations are in the circuits they execute."""

from qiskit.primitives import StatevectorEstimator, BaseSamplerV2 # change: Updated imports to match Qiskit 2.2 API

class LoggingEstimator(StatevectorEstimator): # change: Updated class definition to inherit from StatevectorEstimator
    """An estimator checking what operations were in the circuits it executed."""

    def __init__(self, options=None, operations_callback=None):
        super().__init__(default_precision=0.0, seed=None) # change: Updated super().__init__ call to match StatevectorEstimator
        self.operations_callback = operations_callback

    def _run(self, pubs, **run_options): # change: Updated _run method signature to match StatevectorEstimator
        if self.operations_callback is not None:
            ops = [pub[0].count_ops() for pub in pubs] # change: Updated ops extraction to match new pub format
            self.operations_callback(ops)
        return super()._run(pubs, **run_options) # change: Updated super()._run call to match StatevectorEstimator

class LoggingSampler(BaseSamplerV2): # change: Updated class definition to inherit from BaseSamplerV2
    """A sampler checking what operations were in the circuits it executed."""

    def __init__(self, operations_callback):
        super().__init__() # change: Updated super().__init__ call to match BaseSamplerV2
        self.operations_callback = operations_callback

    def _run(self, pubs, **run_options): # change: Updated _run method signature to match BaseSamplerV2
        ops = [pub[0].count_ops() for pub in pubs] # change: Updated ops extraction to match new pub format
        self.operations_callback(ops)
        return super()._run(pubs, **run_options) # change: Updated super()._run call to match BaseSamplerV2