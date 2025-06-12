# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2025.
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

from ..utils.deprecation import issue_deprecation_msg


class SerializableModelMixin:
    """
    Provides convenient methods for saving and loading models via dill serialization.

    .. warning::
        The legacy :meth:`save` and :meth:`load` methods are deprecated in v0.9.0
        and will be removed in a future release. Please use :meth:`to_dill`
        and :meth:`from_dill` respectively.

    """

    def to_dill(self, file_name: str) -> None:
        """
        Saves this model to the specified file. Internally, the model is serialized via ``dill``.
        All parameters are saved, including a primitive instance that is referenced by internal
        objects. That means if a model is loaded from a file and is used, for instance, for
        inference, the same primitive will be used even if a cloud primitive was used.

        .. warning::
            Replaces the deprecated :meth:`save` method.

        Args:
            file_name: Path where the serialized model will be written.

        Example:
            .. code-block::

                model.to_dill('model_state.dill')
        """
        with open(file_name, "wb") as handler:
            dill.dump(self, handler)

    def save(self, *args) -> None:
        """Backwards compatibility with :meth:`to_dill`, deprecated in v0.9.0."""
        issue_deprecation_msg(
            msg="SerializableModelMixin.save() is deprecated.",
            version="0.9.0",
            remedy="Use the to_dill() method instead.",
            period="4 months",
        )
        self.to_dill(*args)

    @classmethod
    def from_dill(cls, file_name: str) -> Any:
        """
        Loads a model from a file. If the loaded model is not an instance of the class whose
        method was called, then a warning is raised. Nevertheless, the loaded model may be a valid
        model.

        Replaces the deprecated :meth:`load` method.

        Args:
            file_name: Path to the dill file containing the serialized model.

        Returns:
            An instance of the model loaded from disk.

        Example:
            .. code-block::

                loaded = MyModel.from_dill('model_state.dill')

        Raises:
            TypeError: if a loaded model is not an instance of the expected class.
        """
        with open(file_name, "rb") as handler:
            model = dill.load(handler)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is of class {type(model)}. Expected class: {cls}.")
        return model

    @classmethod
    def load(cls, *args) -> Any:
        """Backwards compatibility with :meth:`from_dill`, deprecated in v0.9.0."""
        issue_deprecation_msg(
            msg="SerializableModelMixin.load() is deprecated.",
            version="0.9.0",
            remedy="Use the from_dill() classmethod instead.",
            period="4 months",
        )
        return cls.from_dill(*args)
