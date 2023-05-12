import os
import re

from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes


class CustomCalloutItemDirective(Directive):
    option_spec = {
        "header": directives.unchanged,
        "description": directives.unchanged,
        "button_link": directives.unchanged,
        "button_text": directives.unchanged,
    }

    def run(self):
        try:
            if "description" in self.options:
                description = self.options["description"]
            else:
                description = ""

            if "header" in self.options:
                header = self.options["header"]
            else:
                raise ValueError("header not doc found")

            if "button_link" in self.options:
                button_link = self.options["button_link"]
            else:
                button_link = ""

            if "button_text" in self.options:
                button_text = self.options["button_text"]
            else:
                button_text = ""

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise
            return []

        callout_rst = CALLOUT_TEMPLATE.format(
            description=description, header=header, button_link=button_link, button_text=button_text
        )
        callout_list = StringList(callout_rst.split("\n"))
        callout = nodes.paragraph()
        self.state.nested_parse(callout_list, self.content_offset, callout)
        return [callout]


CALLOUT_TEMPLATE = """
.. raw:: html

    <div class="col-md-6">
        <div class="text-container">
            <h3>{header}</h3>
            <p class="body-paragraph">{description}</p>
            <a class="btn with-right-arrow callout-button" href="{button_link}">{button_text}</a>
        </div>
    </div>
"""
