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

""" Extract deprecation messages from input """

from typing import List
import sys
import os
import argparse


class DeprecationExtractor:
    """Extract deprecation messages"""

    def __init__(self, in_file: str, out_file: str) -> None:
        self._input_filename = in_file
        self._output_filename = out_file
        self._messages = None  # type: List[str]

    def extract_messages(self) -> bool:
        """
        extract deprecation
        Returns:
            bool: if messages were found
        """

        self._messages = None
        messages = set()
        with open(self._input_filename, "rt", encoding="utf8", errors="ignore") as file:
            for line in file:
                if line.find("DeprecationWarning:") > 0:
                    messages.add(line.strip())

        if messages:
            self._messages = sorted(messages)
            return True

        return False

    def save_to_output(self, force_create: bool) -> bool:
        """
        save messages to file if they exist
        Args:
            force_create: create file even if it is empty
        Returns:
            bool: if messages were saved
        """
        if self._output_filename:
            # create file even if it is empty
            if self._messages or force_create:
                with open(self._output_filename, "w", encoding="utf8") as file:
                    if self._messages:
                        file.write("\n".join(self._messages))
                        return True

        return False

    def print_messages(self) -> None:
        """print messages"""
        if self._messages:
            print("---------------------")
            print("Deprecation Messages:")
            print("---------------------")
            for line in self._messages:
                print(line)


def _check_file(path) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"file: '{path}' doesn't exist.")

    return path


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Qiskit Extract Deprecation Messages Tool")
    PARSER.add_argument(
        "-file", type=_check_file, required=True, metavar="file", help="Input file."
    )
    PARSER.add_argument("-output", metavar="output", help="Output file.")

    ARGS = PARSER.parse_args()

    OBJ = DeprecationExtractor(ARGS.file, ARGS.output)
    OBJ.extract_messages()
    OBJ.save_to_output(True)
    OBJ.print_messages()

    sys.exit(0)
