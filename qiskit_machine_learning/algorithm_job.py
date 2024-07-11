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

        For V1 primitives, Qiskit ``PrimitiveJob`` subclassed JobV1 and defined ``submit()``.
        ``PrimitiveJob`` was updated for V2 primitives, no longer subclasses ``JobV1``, and
        now has a private ``_submit()`` method, with ``submit()`` being deprecated as of
        Qiskit version 0.46. This maintains the ``submit()`` for ``AlgorithmJob`` here as
        it's called in many places for such a job. An alternative could be to make
        0.46 the required minimum version and alter all algorithm's call sites to use
        ``_submit()`` and make this an empty class again as it once was. For now this
        way maintains compatibility with the current min version of 0.44.
        """
        # TODO: Considering changing this in the future - see above docstring.
        try:
            super()._submit()
        except AttributeError:
            super().submit()
