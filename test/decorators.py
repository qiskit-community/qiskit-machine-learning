# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Decorator for using with unit tests."""

import functools
import os
import unittest


def slow_test(func):
    """Decorator that signals that the test takes minutes to run.

    Args:
        func (callable): test function to be decorated.

    Returns:
        callable: the decorated function.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        if "run_slow" in os.environ.get("QISKIT_TESTS", ""):
            raise unittest.SkipTest("Skipping slow tests")
        return func(*args, **kwargs)

    return _wrapper
