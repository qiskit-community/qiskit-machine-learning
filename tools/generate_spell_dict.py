# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Generates spelling dictionaries for Sphinx and Pylint and combine them. """

from typing import Set, List
import sys
import os
import argparse
import shutil
import errno
import tempfile
from pathlib import Path
from sphinx.cmd.build import build_main as sphinx_build
from pylint import lint


class SpellDictGenerator:
    """Generates spelling dictionaries for Sphinx and Pylint"""

    _DOCS_DIR = "docs"
    _BUILD_DIR = "_build"
    _STUBS_DIR = "stubs"
    _JUPYTER_EXECUTE_DIR = "jupyter_execute"
    _SPHINX_DICT_FILE = "dummy_spelling_wordlist.txt"
    _SPELLING_SUFFIX = ".spelling"
    _MAKE_FILE = "Makefile"

    def __init__(self, root_dir: str, out_file: str) -> None:
        self._root_dir = root_dir
        self._output_file = out_file
        self._docs_dir = os.path.join(self._root_dir, SpellDictGenerator._DOCS_DIR)
        if not os.path.isdir(self._docs_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self._docs_dir)
        self._build_dir = os.path.join(self._docs_dir, SpellDictGenerator._BUILD_DIR)
        self._stubs_dir = os.path.join(self._docs_dir, SpellDictGenerator._STUBS_DIR)
        self._jupyter_execute_dir = os.path.join(
            self._docs_dir, SpellDictGenerator._JUPYTER_EXECUTE_DIR
        )
        self._sphinx_words: Set[str] = set()
        self._pylint_words: Set[str] = set()

    def generate_sphinx_spell_words(self) -> Set[str]:
        """
        Generates Sphinx spelling dictionary

        Returns:
            spell words
        """
        if os.path.isdir(self._build_dir):
            shutil.rmtree(self._build_dir)
        if os.path.isdir(self._stubs_dir):
            shutil.rmtree(self._stubs_dir)
        if os.path.isdir(self._jupyter_execute_dir):
            shutil.rmtree(self._jupyter_execute_dir)
        try:
            os.mkdir(self._build_dir)
            sphinx_dict_file = os.path.join(self._build_dir, SpellDictGenerator._SPHINX_DICT_FILE)
            # create empty dictionary file
            with open(sphinx_dict_file, "w"):
                pass
            sphinx_build(
                [
                    "-b",
                    "spelling",
                    "-D",
                    f"spelling_word_list_filename={sphinx_dict_file}",
                    self._docs_dir,
                    self._build_dir,
                ]
            )
            self._sphinx_words = SpellDictGenerator._get_sphinx_spell_words(self._build_dir)
            return self._sphinx_words
        finally:
            if os.path.isdir(self._build_dir):
                shutil.rmtree(self._build_dir)
            if os.path.isdir(self._stubs_dir):
                shutil.rmtree(self._stubs_dir)
            if os.path.isdir(self._jupyter_execute_dir):
                shutil.rmtree(self._jupyter_execute_dir)

    @staticmethod
    def _get_sphinx_spell_words(path: str) -> Set[str]:
        words = set()
        for item in os.listdir(path):
            fullpath = os.path.join(path, item)
            file_path = Path(fullpath)
            if file_path.is_dir() and not item.startswith("."):
                word_list = SpellDictGenerator._get_sphinx_spell_words(fullpath)
                words.update(word_list)
            elif file_path.is_file() and file_path.suffix == SpellDictGenerator._SPELLING_SUFFIX:
                word_list = SpellDictGenerator._extract_sphinx_spell_words(fullpath)
                words.update(word_list)

        return words

    @staticmethod
    def _extract_sphinx_spell_words(file_path: str) -> Set[str]:
        words = set()
        with open(file_path, "rt", encoding="utf8") as file:
            for line in file:
                start_idx = line.find("(")
                end_idx = -1
                if start_idx > 0:
                    end_idx = line.find(")", start_idx + 1)
                if start_idx > 0 and end_idx > 0:
                    word = line[start_idx + 1 : end_idx]
                    words.add(word)
        return words

    def generate_pylint_spell_words(self) -> Set[str]:
        """
        Generates Pylint spelling dictionary

        Returns:
            spell words
        Raises:
            FileNotFoundError: makefile not found
            ValueError: Pylint spell not found
        """
        # First read make file to extract pylint options
        make_file = os.path.join(self._root_dir, SpellDictGenerator._MAKE_FILE)
        if not os.path.isfile(make_file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), make_file)
        options = None
        with open(make_file, "rt", encoding="utf8") as file:
            pylint_spell = False
            for line in file:
                if line.startswith("spell:"):
                    pylint_spell = True
                elif pylint_spell and line.find("pylint ") > 0:
                    options = line.split()
                    options = options[1:]
                    break

        if options is None:
            raise ValueError(f"Pylint spell command not found in makefile {make_file}")
        idx = options.index("--spelling-private-dict-file=.pylintdict")
        if idx < 0:
            raise ValueError(f"Pylint spell dict option not found in makefile {make_file}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dict_path = os.path.join(temp_dir, ".pylintdict")
            options[idx] = f"--spelling-private-dict-file={temp_dict_path}"
            options.insert(idx, "--spelling-store-unknown-words=y")
            lint.Run(options, exit=False)
            with open(temp_dict_path, "rt", encoding="utf8") as temp_dict_file:
                words = temp_dict_file.read().splitlines()
                self._pylint_words.update(words)
        return self._pylint_words

    def merge_sort_dict_to_output(self) -> List[str]:
        """Merge and sort Sphinx and Pylint dicts"""
        word_set = set(w.lower() for w in self._sphinx_words)
        word_set.update(w.lower() for w in self._pylint_words)
        words = sorted(word_set)
        with open(self._output_file, "w") as out_file:
            out_file.writelines("%s\n" % word for word in words)
        return words


def check_path(path):
    """valid path argument"""
    if not path or os.path.isdir(path):
        return path

    raise argparse.ArgumentTypeError("readable_dir:{} is not a valid path".format(path))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Qiskit Spelling Dictionary Generation Tool")
    PARSER.add_argument(
        "-path", type=check_path, metavar="path", required=False, help="Root path of project."
    )
    PARSER.add_argument(
        "-output", metavar="output", required=False, default=".pylintdict", help="Output file."
    )

    ARGS = PARSER.parse_args()
    if not ARGS.path:
        ARGS.path = os.getcwd()

    ARGS.path = os.path.abspath(os.path.realpath(os.path.expanduser(ARGS.path)))
    ARGS.output = os.path.join(ARGS.path, ARGS.output)
    OBJ = SpellDictGenerator(ARGS.path, ARGS.output)
    OBJ.generate_sphinx_spell_words()
    OBJ.generate_pylint_spell_words()
    OBJ.merge_sort_dict_to_output()

    sys.exit(os.EX_OK)
