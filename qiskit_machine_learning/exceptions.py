# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Machine Learning Exception """

from qiskit.exceptions import QiskitError


class QiskitMachineLearningError(QiskitError):
    """Class for errors returned by Qiskit Machine Learning module."""

    pass


class QiskitMachineLearningWarning(UserWarning):
    """Class for warning returned by Qiskit Machine Learning module."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class AlgorithmError(QiskitError):
    """For Algorithm specific errors."""

    pass
