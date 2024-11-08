# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Deprecation utilities"""

from typing import Callable, Any
import functools
import warnings


def deprecate_function(deprecated: str, version: str, remedy: str, stacklevel: int = 2) -> Callable:
    """Emit a warning prior to calling decorated function.
    Args:
        deprecated: Function being deprecated.
        version: First release the function is deprecated.
        remedy: User action to take.
        stacklevel: The warning stack-level to use.

    Returns:
        The decorated, deprecated callable.
    """

    def decorator(func: Callable) -> Callable:
        """Emit a deprecation warning."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Callable:
            """Emit a deprecation warning."""
            issue_deprecation_msg(
                f"The {deprecated} method is deprecated",
                version,
                remedy,
                stacklevel + 1,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_arguments(deprecated: str, version: str, remedy: str, stacklevel: int = 2) -> None:
    """Emit a warning about deprecated keyword arguments.

    Args:
        deprecated: Keyword arguments being deprecated.
        version: First release the function is deprecated.
        remedy: User action to take.
        stacklevel: The warning stack-level to use.
    """
    issue_deprecation_msg(
        f"The '{deprecated}' keyword arguments are deprecated",
        version,
        remedy,
        stacklevel + 1,
    )


def issue_deprecation_msg(
    msg: str, version: str, remedy: str, stacklevel: int = 2, period: str = "3 months"
) -> None:
    """Emit a deprecation warning.

    Args:
        msg: Deprecation message.
        version: First release the function is deprecated.
        remedy: User action to take.
        stacklevel: The warning stack-level to use.
        period: Deprecation period.
    """
    warnings.warn(
        f"{msg} as of qiskit-machine-learning {version} "
        f"and will be removed no sooner than {period} after the release date. {remedy}",
        DeprecationWarning,
        stacklevel=stacklevel + 1,  # Increment to account for this function.
    )
