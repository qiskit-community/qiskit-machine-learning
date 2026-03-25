#!/usr/bin/env python3
# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to verify qiskit copyright file headers"""

import argparse
import multiprocessing
import os
import sys
import re

# regex for character encoding from PEP 263
pep263 = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")


def discover_files(code_paths, exclude_folders):
    """Find all .py, .pyx, .pxd files in a list of trees"""
    out_paths = []
    for path in code_paths:
        if os.path.isfile(path):
            out_paths.append(path)
        else:
            for directory in os.walk(path):
                dir_path = directory[0]
                for folder in exclude_folders:
                    if folder in directory[1]:
                        directory[1].remove(folder)
                for subfile in directory[2]:
                    if (
                        subfile.endswith(".py")
                        or subfile.endswith(".pyx")
                        or subfile.endswith(".pxd")
                    ):
                        out_paths.append(os.path.join(dir_path, subfile))
    return out_paths


def get_years(line):
    """Return first and last year for copyright line"""
    curr_years = []
    for word in line.strip().split():
        for year in word.strip().split(","):
            if year.startswith("20") and len(year) >= 4:
                try:
                    curr_years.append(int(year[0:4]))
                except ValueError:
                    pass
    if len(curr_years) > 1:
        return curr_years[0], curr_years[1]
    if len(curr_years) == 1:
        return curr_years[0], curr_years[0]
    return None, None


def validate_header(file_path):
    """Validate the header for a single file"""
    header = """# This code is part of a Qiskit project.
#
"""
    apache_text = """#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""

    with open(file_path, encoding="utf8") as code_file:
        lines = code_file.readlines()

    stfc_year = 2024
    count = 0
    start = 0
    reason = None
    passed = True
    for index, line in enumerate(lines):
        count += 1
        if count > 5:
            reason = "Header not found in first 5 lines"
            passed = False
        if count <= 2 and pep263.match(line):
            reason = "Unnecessary encoding specification (PEP 263, 3120)"
            passed = False
        if passed and line == "# This code is part of a Qiskit project.\n":
            start = index
            break

    if passed and "".join(lines[start : start + 2]) != header:
        reason = f"Header up to copyright line does not match: {header}"
        passed = False

    if passed and not lines[start + 2].startswith("# (C) Copyright IBM 20"):
        reason = "Header IBM copyright line not found"
        passed = False

    _, ibm_last_year = get_years(lines[start + 2])
    if passed and ibm_last_year >= stfc_year:
        if not lines[start + 3].startswith("# (C) Copyright UKRI-STFC (Hartree Centre) 20"):
            reason = "Header STFC copyright line not found"
            passed = False

        if passed and "".join(lines[start + 4 : start + 12]) != apache_text:
            reason = f"Header apache text string doesn't match:\n {apache_text}"
            passed = False
    elif passed and ibm_last_year < stfc_year:
        if "".join(lines[start + 3 : start + 11]) != apache_text:
            reason = f"Header apache text string doesn't match:\n {apache_text}"
            passed = False

    return file_path, passed, reason


def _main():
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qiskit_machine_learning"
    )
    parser = argparse.ArgumentParser(description="Check file headers.")
    parser.add_argument(
        "paths",
        type=str,
        nargs="*",
        default=[default_path],
        help="Paths to scan by default uses ../qiskit_machine_learning from the script",
    )
    args = parser.parse_args()
    files = discover_files(args.paths, exclude_folders=[])
    with multiprocessing.Pool() as pool:
        res = pool.map(validate_header, files)
    failed_files = [x for x in res if x[1] is False]
    if len(failed_files) > 0:
        for failed_file in failed_files:
            sys.stderr.write(f"{failed_file[0]} failed header check because:\n")
            sys.stderr.write(f"{failed_file[2]}\n\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    _main()
