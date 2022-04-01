# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A mixin class for saving and loading models."""

import logging
from typing import Any

import dill


logger = logging.getLogger(__name__)


class SerializableModelMixin:
    """
    This class provides convenient methods for saving and loading models.
    """

    def save(self, file_name: str) -> None:
        """
        Saves this model to the file. Internally, the model is serialized via ``dill`` and saved to
        the specified file. All parameters are saved, including a quantum instance that is
        referenced by internal objects. That means if a model is loaded from a model and is used
        the same quantum instance and a corresponding backend will be used.

        Args:
            file_name: a file name or path where to save the model.
        """
        with open(file_name, "wb") as handler:
            dill.dump(self, handler)

    @classmethod
    def load(cls, file_name: str) -> Any:
        """
        Loads a model from the file. If the loaded model is not an instance of the class whose
        method was called, then a warning is raised. Nevertheless, the loaded model may be a valid
        model.

        Args:
            file_name: a file name or path to load a model from.

        Returns:
            A loaded model.
        """
        with open(file_name, "rb") as handler:
            model = dill.load(handler)
        if not isinstance(model, cls):
            logger.warning(
                "Loaded a model of a different class. Expected class: %s, loaded: %s.",
                cls,
                type(model),
            )
        return model