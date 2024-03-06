#!/usr/bin/env python3
# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to find gpu decorated test methods"""

import os
import builtins
import unittest
import sys
import argparse

_TEST = "test"
_DECORATOR = "_decorator"


def _find_methods(suite):
    """find gpu test decorated methods recursively"""
    # pylint: disable=no-name-in-module
    from test import gpu

    methods = []
    for obj in suite:
        if isinstance(obj, unittest.loader._FailedTest):
            raise obj._exception
        if isinstance(obj, unittest.TestSuite):
            methods.extend(_find_methods(obj))
        elif isinstance(obj, unittest.TestCase):
            test_method = getattr(obj, obj._testMethodName)
            decorator = getattr(obj.setUp, _DECORATOR, None)
            if decorator is None:
                decorator = getattr(test_method, _DECORATOR, None)
            if decorator is not None and decorator is gpu:
                cls_str = f"{obj.__module__}.{obj.__class__.__name__}"
                methods.append(f"{_TEST}.{cls_str}.{test_method.__name__}")
        else:
            raise builtins.Exception(f"Unexpected class {type(obj)}")
    return methods


def _save_methods_to_file(path):
    """save list of methods to file"""
    suite = unittest.TestLoader().discover(_TEST, pattern="test*.py", top_level_dir=_TEST)
    methods = _find_methods(suite)
    if len(methods) == 0:
        raise builtins.Exception("No GPU decorated test methods found.")
    with open(path, "w", encoding="utf8") as textfile:
        textfile.write("\n".join(methods))


def _check_path(path):
    """valid path argument"""
    if path and os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"path:{path} is not a valid file path")
    return path


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Find GPU Unit Test Methods")
    PARSER.add_argument("-path", type=_check_path, metavar="path", help="File path to save list.")

    ARGS = PARSER.parse_args()
    sys.path.insert(0, os.path.abspath("."))
    _save_methods_to_file(ARGS.path)
