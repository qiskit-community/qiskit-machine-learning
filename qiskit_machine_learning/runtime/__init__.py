# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Deprecation: Runtime (:mod:`qiskit_machine_learning.runtime`)
=============================================================

.. currentmodule:: qiskit_machine_learning.runtime

Programs that embed Qiskit Runtime in the algorithmic interfaces and facilitate usage of
algorithms and scripts in the cloud.


.. autosummary::
   :toctree: ../stubs/
   :nosignatures:


   TorchRuntimeClient
   TorchRuntimeResult
"""

from .hookbase import HookBase
from .torch_runtime_client import TorchRuntimeClient, TorchRuntimeResult, obj_to_str, str_to_obj


__all__ = ["HookBase", "TorchRuntimeClient", "TorchRuntimeResult", "obj_to_str", "str_to_obj"]
