# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base class for handling communication with program users."""

import json, sys
from typing import Any, Type

sys.path.append("..")
from qiskit_machine_learning.runtime.mock_run_utils.runtime_utils import RuntimeEncoder


class UserMessenger:
    """Base class for handling communication with program users.
    This class can be used when writing a new Qiskit Runtime program.
    """

    def publish(
        self, message: Any, encoder: Type[json.JSONEncoder] = RuntimeEncoder, final: bool = False
    ) -> None:
        """Publish message.
        You can use this method to publish messages, such as interim and final results,
        to the program user. The messages will be made immediately available to the user,
        but they may choose not to receive the messages.
        The `final` parameter is used to indicate whether the message is
        the final result of the program. Final results may be processed differently
        from interim results.
        Args:
            message: Message to be published. Can be any type.
            encoder: An optional JSON encoder for serializing
            final: Whether the message being published is the final result.
        """
        # pylint: disable=unused-argument
        # Default implementation for testing.
        print(json.dumps(message, cls=encoder))
