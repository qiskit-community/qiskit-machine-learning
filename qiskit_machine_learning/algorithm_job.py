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

"""
AlgorithmJob class
"""
from qiskit.primitives.primitive_job import PrimitiveJob


class AlgorithmJob(PrimitiveJob):
    """
    This class is introduced for typing purposes and provides no
    additional function beyond that inherited from its parents.

    Update: :meth:`AlgorithmJob.submit()` method added. See its
    documentation for more info.
    """

    def submit(self) -> None:
        """
        Submit the job for execution.

        Since the library has been migrated to Qiskit v2.1, it is no longer necessary to
        keep the :meth:``JobV1.submit()`` for the exception handling.
        """
        super()._submit()
