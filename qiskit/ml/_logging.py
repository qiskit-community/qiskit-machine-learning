# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for Machine Learning logging."""

from typing import Optional
from qiskit.aqua import (QiskitLogDomains,
                         get_logging_level,
                         set_logging_level)


def get_qiskit_ml_logging() -> int:
    """
    Returns the current Qiskit ML logging level

    Returns:
        int: logging level
    """
    return get_logging_level(QiskitLogDomains.DOMAIN_ML)


def set_qiskit_ml_logging(level: int, filepath: Optional[str] = None) -> None:
    """
    Updates the Qiskit ML logging with the appropriate logging level

    Args:
        level: minimum severity of the messages that are displayed.
        filepath: file to receive logging data
    """
    set_logging_level(level, [QiskitLogDomains.DOMAIN_ML], filepath)
