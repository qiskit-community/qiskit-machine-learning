#!/bin/bash

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

# Script to ignore untagged release notes prior to deploying docs to site.

LATEST_TAG=$(git describe --tags --abbrev=0)

for file_changed in `git diff --name-only HEAD $LATEST_TAG`
do
    if [[ $file_changed == releasenotes/notes/* ]]; then
        isInFile=$(grep -Exq "\s*$file_changed," docs/release_notes.rst >/dev/null; echo $?)
        if [ $isInFile -ne 0 ]; then
            isInFile=$(grep -Exq "\s*:ignore-notes:\s*" docs/release_notes.rst >/dev/null; echo $?)
            if [ $isInFile -ne 0 ]; then
                echo "   :ignore-notes:" >> docs/release_notes.rst
            fi
            echo "Release note changed since $LATEST_TAG: $file_changed. Ignore in docs/release_notes.rst"
            echo "     $file_changed," >> docs/release_notes.rst
        fi
    fi
done

echo "Contents of docs/release_notes.rst:"
echo "$(cat docs/release_notes.rst)"

exit 0