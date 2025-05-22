r"""Extension for signatures.

Essentially just an admonition box with no body, only a title.
"""

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class Signature(Directive):
    """Essentially just an admonition box with no body, only a title.

    References
    ----------
    - https://docutils.sourceforge.io/docutils/parsers/rst/directives/admonitions.py
    """

    required_arguments = 0
    r"""The number of required arguments (default: 0)."""
    optional_arguments = 0
    r"""The number of optional arguments (default: 0)."""
    final_argument_whitespace = True
    r"""A boolean, indicating if the final argument may contain whitespace."""
    has_content = True
    r"""A boolean; True if content is allowed.  Client code must handle the case where
    content is required but not supplied (an empty content list will be supplied)."""

    def run(self) -> list[nodes.Node]:
        r"""Run the directive."""
        # Raise an error if the directive does not have contents.
        self.assert_has_content()

        if not len(self.content) == 1:
            raise ValueError("Signature directive must have exactly one argument.")

        text = "\n".join(self.content)
        admonition_node = nodes.admonition(text, **self.options)
        self.add_name(admonition_node)

        title_text = self.content[0]
        signature_text = f"Signature: {title_text}"

        textnodes, messages = self.state.inline_text(signature_text, self.lineno)
        title = nodes.title(signature_text, "", *textnodes)
        title.source, title.line = self.state_machine.get_source_and_line(self.lineno)
        admonition_node += title
        admonition_node += messages

        if "classes" not in self.options:
            admonition_node["classes"] += ["admonition-" + nodes.make_id("signature")]

        return [admonition_node]


def setup(app: Sphinx) -> dict:
    r"""Install the extension."""
    app.add_directive("signature", Signature)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
