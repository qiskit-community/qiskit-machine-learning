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

import os
import re
import subprocess
import tempfile
from functools import partial

from docutils import nodes
from docutils.parsers.rst.directives.tables import Table
from docutils.parsers.rst import Directive, directives
from sphinx.util import logging


logger = logging.getLogger(__name__)

translations_list = [
    ("en", "English"),
    ("bn_BN", "Bengali"),
    ("fr_FR", "French"),
    ("hi_IN", "Hindi"),
    ("it_IT", "Italian"),
    ("ja_JP", "Japanese"),
    ("ko_KR", "Korean"),
    ("ru_RU", "Russian"),
    ("es_UN", "Spanish"),
    ("ta_IN", "Tamil"),
    ("tr_TR", "Turkish"),
    ("vi_VN", "Vietnamese"),
]

default_language = "en"


def setup(app):
    app.connect("config-inited", _extend_html_context)
    app.add_config_value("content_prefix", "documentation/machine-learning", "")
    app.add_config_value("translations", True, "html")


def _extend_html_context(app, config):
    context = config.html_context
    context["translations"] = config.translations
    context["translations_list"] = translations_list
    context["current_translation"] = _get_current_translation(config) or config.language
    context["translation_url"] = partial(_get_translation_url, config)
    context["version_label"] = _get_version_label(config)


def _get_current_translation(config):
    language = config.language or default_language
    try:
        found = next(v for k, v in translations_list if k == language)
    except StopIteration:
        found = None
    return found


def _get_translation_url(config, code, pagename):
    base = "/locale/%s" % code if code and code != default_language else ""
    return _get_url(config, base, pagename)


def _get_version_label(config):
    return "%s" % (_get_current_translation(config) or config.language,)


def _get_url(config, base, pagename):
    return _add_content_prefix(config, "%s/%s.html" % (base, pagename))


def _add_content_prefix(config, url):
    prefix = ""
    if config.content_prefix:
        prefix = "/%s" % config.content_prefix
    return "%s%s" % (prefix, url)
