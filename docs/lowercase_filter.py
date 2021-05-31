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

""" Implements a Lower Case Filter for Sphinx spelling """

from enchant import tokenize


class LowercaseFilter(tokenize.Filter):
    """ Lower Case Filter """

    def _split(self, word):
        """Filter method for sub-tokenization of tokens.

        This method must be a tokenization function that will split the
        given word into sub-tokens according to the needs of the filter.
        The default behavior is not to split any words.
        """
        # Don't split, just lower case to test against lowercase dict
        return super()._split(word.lower())
