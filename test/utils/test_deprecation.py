# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for the deprecation utilities."""

from test import QiskitMachineLearningTestCase

from qiskit_machine_learning.utils.deprecation import (
    deprecate_arguments,
    deprecate_function,
    issue_deprecation_msg,
)


class TestDeprecation(QiskitMachineLearningTestCase):
    """Tests for the deprecation utility functions."""

    def test_issue_deprecation_msg(self):
        """Test that issue_deprecation_msg emits a well-formed DeprecationWarning."""
        with self.assertWarns(DeprecationWarning) as ctx:
            issue_deprecation_msg(
                msg="Some feature is deprecated",
                version="0.10.0",
                remedy="Use the new feature instead.",
                period="6 months",
            )
        message = str(ctx.warning)
        self.assertIn("Some feature is deprecated", message)
        self.assertIn("0.10.0", message)
        self.assertIn("6 months", message)
        self.assertIn("Use the new feature instead.", message)

    def test_issue_deprecation_msg_default_period(self):
        """Test the default deprecation period is used when not provided."""
        with self.assertWarns(DeprecationWarning) as ctx:
            issue_deprecation_msg(
                msg="Another feature is deprecated",
                version="0.11.0",
                remedy="Do something else.",
            )
        self.assertIn("3 months", str(ctx.warning))

    def test_deprecate_function_decorator(self):
        """Test that deprecate_function wraps a function and warns before calling it."""

        @deprecate_function(
            deprecated="my_func", version="0.10.0", remedy="Use new_func() instead."
        )
        def my_func(a, b):
            """A function to be deprecated."""
            return a + b

        with self.assertWarns(DeprecationWarning) as ctx:
            result = my_func(2, 3)

        self.assertEqual(result, 5)
        message = str(ctx.warning)
        self.assertIn("my_func", message)
        self.assertIn("0.10.0", message)
        self.assertIn("Use new_func() instead.", message)

    def test_deprecate_function_preserves_metadata(self):
        """Test that deprecate_function preserves the wrapped function's metadata."""

        @deprecate_function(deprecated="named_func", version="0.10.0", remedy="Use x instead.")
        def named_func():
            """Docstring for named_func."""

        self.assertEqual(named_func.__name__, "named_func")
        self.assertEqual(named_func.__doc__, "Docstring for named_func.")

    def test_deprecate_arguments(self):
        """Test that deprecate_arguments emits a DeprecationWarning about kwargs."""
        with self.assertWarns(DeprecationWarning) as ctx:
            deprecate_arguments(
                deprecated="old_kwarg",
                version="0.10.0",
                remedy="Use new_kwarg instead.",
            )
        message = str(ctx.warning)
        self.assertIn("old_kwarg", message)
        self.assertIn("0.10.0", message)
        self.assertIn("Use new_kwarg instead.", message)
