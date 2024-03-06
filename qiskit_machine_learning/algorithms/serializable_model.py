# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A mixin class for saving and loading models."""

from typing import Any

import dill


class SerializableModelMixin:
    """
    Provides convenient methods for saving and loading models.
    """

    def save(self, file_name: str) -> None:
        """
        Saves this model to the specified file. Internally, the model is serialized via ``dill``.
        All parameters are saved, including a primitive instance that is referenced by internal
        objects. That means if a model is loaded from a file and is used, for instance, for
        inference, the same primitive will be used even if a cloud primitive was used.

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

        Raises:
            TypeError: if a loaded model is not an instance of the expected class.
        """
        with open(file_name, "rb") as handler:
            model = dill.load(handler)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is of class {type(model)}. Expected class: {cls}.")
        return model
