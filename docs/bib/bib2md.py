"""Render bibtex file as markdown with keys as headings that can be cross-referenced."""

from pathlib import Path

import pybtex.backends.markdown as fmt
from pybtex.database import parse_string
from pybtex.richtext import Tag, Text
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.formatting.unsrt import field, sentence

fmt.SPECIAL_CHARS.remove("[")
fmt.SPECIAL_CHARS.remove("]")
fmt.SPECIAL_CHARS.remove("(")
fmt.SPECIAL_CHARS.remove(")")
fmt.SPECIAL_CHARS.remove("-")

HERE = Path(__file__).parent


class MyStyle(UnsrtStyle):
    # default_sorting_style = "author_year_title"
    def format_title(self, e, which_field, as_sentence=True):
        formatted_title = field(
            which_field,
            apply_func=lambda text: Tag("tt", Text('"', text.capitalize(), '"')),
        )
        if as_sentence:
            return sentence[formatted_title]
        else:
            return formatted_title


bib = parse_string((HERE / "refs.bib").read_text(), "bibtex")

formatted = MyStyle().format_entries(bib.entries.values())

md = ""
for entry in formatted:
    md += f"### `{entry.key}`\n\n"
    md += entry.text.render_as("markdown") + "\n\n"

(HERE.parent / "references.md").write_text(md)
