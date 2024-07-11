# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
utils.algorithm_globals
=======================
Common (global) properties used across qiskit_algorithms.

.. currentmodule:: qiskit_algorithms.utils.algorithm_globals

Includes:

  * Random number generator and random seed.

    Algorithms can use the generator for random values, as needed, and it
    can be seeded here for reproducible results when using such an algorithm.
    This is often important, for example in unit tests, where the same
    outcome is desired each time (reproducible) and not have it be variable
    due to randomness.

Attributes:
    random_seed (int | None): Random generator seed (read/write).
    random (np.random.Generator): Random generator (read-only)
"""

from __future__ import annotations

import warnings

import numpy as np


class QiskitAlgorithmGlobals:
    """Global properties for algorithms."""

    # The code is done to work even after some future removal of algorithm_globals
    # from Qiskit (qiskit.utils). All that is needed in the future, after that, if
    # this is updated, is just the logic in the except blocks.
    #
    # If the Qiskit version exists this acts a redirect to that (it delegates the
    # calls off to it). In the future when that does not exist this has similar code
    # in the except blocks here, as noted above, that will take over. By delegating
    # to the Qiskit instance it means that any existing code that uses that continues
    # to work. Logic here in qiskit_algorithms though uses this instance and the
    # random check here has logic to warn if the seed here is not the same as the Qiskit
    # version so we can detect direct usage of the Qiskit version and alert the user to
    # change their code to use this. So simply changing from:
    #     from qiskit.utils import algorithm_globals
    # to
    #     from qiskit_algorithm.utils import algorithm_globals

    def __init__(self) -> None:
        self._random_seed: int | None = None
        self._random: np.random.Generator | None = None

    @property
    def random_seed(self) -> int | None:
        """Random seed property (getter/setter)."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)

                from qiskit.utils import algorithm_globals as qiskit_globals

                return qiskit_globals.random_seed

        except ImportError:
            return self._random_seed

    @random_seed.setter
    def random_seed(self, seed: int | None) -> None:
        """Set the random generator seed.

        Args:
            seed: If ``None`` then internally a random value is used as a seed
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)

                from qiskit.utils import algorithm_globals as qiskit_globals

                qiskit_globals.random_seed = seed
                # Mirror the seed here when set via this random_seed. If the seed is
                # set on the qiskit.utils instance then we can detect it's different
                self._random_seed = seed

        except ImportError:
            self._random_seed = seed
            self._random = None

    @property
    def random(self) -> np.random.Generator:
        """Return a numpy np.random.Generator (default_rng) using random_seed."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)

                from qiskit.utils import algorithm_globals as qiskit_globals

                if self._random_seed != qiskit_globals.random_seed:
                    # If the seeds are different - likely this local is None and the qiskit.utils
                    # algorithms global was seeded directly then we will warn to use this here as
                    # the Qiskit version is planned to be removed in a future version of Qiskit.
                    warnings.warn(
                        "Using random that is seeded via qiskit.utils algorithm_globals is deprecated "
                        "since version 0.2.0. Instead set random_seed directly to "
                        "qiskit_algorithms.utils algorithm_globals.",
                        category=DeprecationWarning,
                        stacklevel=2,
                    )

                return qiskit_globals.random

        except ImportError:
            if self._random is None:
                self._random = np.random.default_rng(self._random_seed)
            return self._random


# Global instance to be used as the entry point for globals.
algorithm_globals = QiskitAlgorithmGlobals()
