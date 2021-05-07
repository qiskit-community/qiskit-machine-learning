# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Machine Learning Test Case"""

from typing import Optional
from abc import ABC
import warnings
import inspect
import logging
import os
import unittest
import time
from qiskit.exceptions import MissingOptionalLibraryError

# disable deprecation warnings that can cause log output overflow
# pylint: disable=unused-argument


def _noop(*args, **kargs):
    pass


# disable warning messages
# warnings.warn = _noop


def requires_extra_library(test_item):
    """Decorator that skips test if an extra library is not available

    Args:
        test_item (callable): function to be decorated.

    Returns:
        callable: the decorated function.
    """

    def wrapper(self, *args):
        try:
            test_item(self, *args)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        return wrapper

    return wrapper


class QiskitMachineLearningTestCase(unittest.TestCase, ABC):
    """Machine Learning Test Case"""

    moduleName = None
    log = None

    def setUp(self) -> None:
        warnings.filterwarnings("default", category=DeprecationWarning)
        self._started_at = time.time()
        self._class_location = __file__

    def tearDown(self) -> None:
        elapsed = time.time() - self._started_at
        if elapsed > 5.0:
            print("({:.2f}s)".format(round(elapsed, 2)), flush=True)

    @classmethod
    def setUpClass(cls) -> None:
        cls.moduleName = os.path.splitext(inspect.getfile(cls))[0]
        cls.log = logging.getLogger(cls.__name__)

        # Set logging to file and stdout if the LOG_LEVEL environment variable
        # is set.
        if os.getenv("LOG_LEVEL"):
            # Set up formatter.
            log_fmt = "{}.%(funcName)s:%(levelname)s:%(asctime)s:" " %(message)s".format(
                cls.__name__
            )
            formatter = logging.Formatter(log_fmt)

            # Set up the file handler.
            log_file_name = "%s.log" % cls.moduleName
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(formatter)
            cls.log.addHandler(file_handler)

            # Set the logging level from the environment variable, defaulting
            # to INFO if it is not a valid level.
            level = logging._nameToLevel.get(os.getenv("LOG_LEVEL"), logging.INFO)
            cls.log.setLevel(level)

    def get_resource_path(self, filename: str, path: Optional[str] = None) -> str:
        """Get the absolute path to a resource.
        Args:
            filename: filename or relative path to the resource.
            path: path used as relative to the filename.
        Returns:
            str: the absolute path to the resource.
        """
        root = os.path.dirname(self._class_location)
        path = root if path is None else os.path.join(root, path)
        return os.path.normpath(os.path.join(path, filename))
