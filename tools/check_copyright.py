# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Check copyright year """

from typing import Tuple, Union, List
import sys
import os
import datetime
import argparse
import subprocess
import traceback


class CopyrightChecker:
    """Check copyright"""

    _UTF_STRING = "# -*- coding: utf-8 -*-"
    _COPYRIGHT_STRING = "# (C) Copyright IBM "

    def __init__(self, root_dir: str) -> None:
        self._root_dir = root_dir
        self._current_year = datetime.datetime.now().year
        self._changed_files = self._get_changed_files()

    @staticmethod
    def _exception_to_string(excp: Exception) -> str:
        stack = traceback.extract_stack()[:-3] + traceback.extract_tb(excp.__traceback__)
        pretty = traceback.format_list(stack)
        return "".join(pretty) + f"\n  {excp.__class__} {excp}"

    @staticmethod
    def _get_year_from_date(date) -> int:
        if not date or len(date) < 4:
            return None

        return int(date[:4])

    def _cmd_execute(self, args: List[str]) -> Tuple[str, Union[None, str]]:
        # execute command
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        with subprocess.Popen(
            args,
            cwd=self._root_dir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as popen:
            out, err = popen.communicate()
            popen.wait()
            out_str = out.decode("utf-8").strip()
            err_str = err.decode("utf-8").strip()
            err_str = err_str if err_str else None
            return out_str, err_str

    def _get_changed_files(self) -> List[str]:
        out_str, err_str = self._cmd_execute(["git", "diff", "--name-only", "HEAD"])
        if err_str:
            raise Exception(err_str)

        return out_str.splitlines()

    def _get_file_last_year(self, relative_path: str) -> int:
        last_year = None
        errors = []
        try:
            out_str, err_str = self._cmd_execute(
                ["git", "log", "-1", "--format=%cI", relative_path]
            )
            last_year = CopyrightChecker._get_year_from_date(out_str)
            if err_str:
                errors.append(err_str)
        except Exception as ex:  # pylint: disable=broad-except
            errors.append(f"'{relative_path}' Last year: {str(ex)}")

        if errors:
            raise ValueError(" - ".join(errors))

        return last_year

    def check_copyright(self, file_path) -> Tuple[bool, bool, bool]:
        """check copyright for a file"""
        file_with_utf8 = False
        file_with_invalid_year = False
        file_has_header = False
        try:
            with open(file_path, "rt", encoding="utf8") as file:
                for line in file:
                    relative_path = os.path.relpath(file_path, self._root_dir)
                    if line.startswith(CopyrightChecker._UTF_STRING):
                        print(f"File contains utf-8 header: '{relative_path}'")
                        file_with_utf8 = True

                    if not line.startswith(CopyrightChecker._COPYRIGHT_STRING):
                        continue

                    file_has_header = True
                    curr_years = []
                    for word in line.strip().split():
                        for year in word.strip().split(","):
                            if year.startswith("20") and len(year) >= 4:
                                try:
                                    curr_years.append(int(year[0:4]))
                                except ValueError:
                                    pass

                    header_start_year = None
                    header_last_year = None
                    if len(curr_years) > 1:
                        header_start_year = curr_years[0]
                        header_last_year = curr_years[1]
                    elif len(curr_years) == 1:
                        header_start_year = header_last_year = curr_years[0]

                    if relative_path in self._changed_files:
                        self._changed_files.remove(relative_path)
                        last_year = self._current_year
                    else:
                        last_year = self._get_file_last_year(relative_path)
                    if last_year and header_last_year != last_year:
                        new_line = "# (C) Copyright IBM "
                        if header_start_year and header_start_year != last_year:
                            new_line += f"{header_start_year}, "

                        new_line += f"{self._current_year}.\n"
                        print(
                            f"Wrong Copyright Year:'{relative_path}': ",
                            f"Current:'{line[:-1]}' Correct:'{new_line[:-1]}'",
                        )
                        file_with_invalid_year = True

                    break

        except UnicodeDecodeError:
            return file_with_utf8, file_with_invalid_year, file_has_header

        return file_with_utf8, file_with_invalid_year, file_has_header

    def check(self) -> Tuple[int, int, int]:
        """check copyright"""
        return self._check(self._root_dir)

    def _check(self, path: str) -> Tuple[int, int, int]:
        files_with_utf8 = 0
        files_with_invalid_year = 0
        files_with_header = 0
        for item in os.listdir(path):
            fullpath = os.path.join(path, item)
            if os.path.isdir(fullpath):
                if not item.startswith("."):
                    files = self._check(fullpath)
                    files_with_utf8 += files[0]
                    files_with_invalid_year += files[1]
                    files_with_header += files[2]
                continue

            if os.path.isfile(fullpath):
                # check copyright year
                (
                    file_with_utf8,
                    file_with_invalid_year,
                    file_has_header,
                ) = self.check_copyright(fullpath)
                if file_with_utf8:
                    files_with_utf8 += 1
                if file_with_invalid_year:
                    files_with_invalid_year += 1
                if file_has_header:
                    files_with_header += 1

        return files_with_utf8, files_with_invalid_year, files_with_header


def check_path(path):
    """valid path argument"""
    if not path or os.path.isdir(path):
        return path

    raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Check Copyright Tool")
    PARSER.add_argument("-path", type=check_path, metavar="path", help="Root path of project.")

    ARGS = PARSER.parse_args()
    if not ARGS.path:
        ARGS.path = os.getcwd()

    ARGS.path = os.path.abspath(os.path.realpath(os.path.expanduser(ARGS.path)))
    INVALID_UTF8, INVALID_YEAR, HAS_HEADER = CopyrightChecker(ARGS.path).check()
    print(f"{INVALID_UTF8} files have utf8 headers.")
    print(f"{INVALID_YEAR} of {HAS_HEADER} files with copyright header have wrong years.")

    sys.exit(0 if INVALID_UTF8 == 0 and INVALID_YEAR == 0 else 1)
