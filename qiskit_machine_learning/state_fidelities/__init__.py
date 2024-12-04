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

r"""
State Fidelities (:mod:`qiskit_machine_learning.state_fidelities`)
==================================================================

Algorithms that compute the fidelity of two given quantum states.

.. currentmodule:: qiskit_machine_learning.state_fidelities

State fidelities
----------------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseStateFidelity
   ComputeUncompute

Results
-------

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    StateFidelityResult

"""

from .base_state_fidelity import BaseStateFidelity
from .compute_uncompute import ComputeUncompute
from .state_fidelity_result import StateFidelityResult

__all__ = ["BaseStateFidelity", "ComputeUncompute", "StateFidelityResult"]
