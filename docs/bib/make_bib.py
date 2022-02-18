#!/usr/bin/env python

"""Make bib.py from refs.bib.

Usage:

- Put the references you refer to in dapper's docstrings in refs.bib.
  Consider using `copy_refs.py` for this purpose.
- Run this to update the bib.py file.
- Run pdoc.

Note: requires pandoc, and presumably latex too.
"""

from pathlib import Path
from subprocess import run
from textwrap import dedent
import os

os.chdir(Path(__file__).parent)


# Create .md doc with \notice{*}
open("bib.md", "w").write(
    r"""
---
bibliography: refs.bib
nocite: '@*'
...

# Bibliography
"""
)


# Convert to .rst (to process the references)
# Requires recent version of pandoc
# NB: Unfortunately, converting directly to .md
#     outputs in a verbose "list-ish" format.
# Note: for CSL styles see https://editor.citationstyles.org/about/
run(
    [
        "pandoc",
        "--citeproc",
        "--csl=data-science-journal.csl",
        "-s",
        "bib.md",
        "-o",
        "bib.rst",
    ],
    check=True,
)


# Parse rst
rst = open("bib.rst").readlines()
linenumbers = []
REF_START = "      :name: ref-"
# Get ref names and line numbers
for lineno, ln in enumerate(rst):
    if REF_START in ln:
        name = ln[len(REF_START) :].strip()
        linenumbers.append((name, lineno))
# Get ref blocks
ref_dict = {}
for i, (name, lineno1) in enumerate(linenumbers):
    try:
        lineno2 = linenumbers[i + 1][1] - 1
    except IndexError:
        lineno2 = len(rst)
    block = rst[lineno1 + 2 : lineno2]
    block = dedent("".join(block)).strip()
    ref_dict[name] = block


# Clean up
Path("bib.rst").unlink()
Path("bib.md").unlink()

# Sort
# ref_dict = dict(sorted(ref_dict.items()))

# Write bib.py
with open("bib.py", "w") as bibfile:

    def _print(*a, **b):
        print(*a, **b, file=bibfile)

    _print('"""Bibliography/references."""')
    for key, block in ref_dict.items():
        _print("")
        _print(key, "=", "None")
        _print('"""')
        _print(block)
        _print('"""')
