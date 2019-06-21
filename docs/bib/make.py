#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from subprocess import check_output
from contextlib import contextmanager
import re

@contextmanager
def rewrite(fname):
    """Work with lines in-place (read, yield, write)."""
    with open(fname, 'r') as f: lines = [line for line in f]
    yield lines
    with open(fname, 'w') as f: f.write("".join(lines))


# Clean dir
out = check_output(["find", ".", "-type", "f",
  "!", "-name", "*.tex", "-name", "bib.*", "-delete"])
# Extract refs
out = check_output(["reference_strip","bib.tex",
  "/Users/pataan/Dropbox/DPhil/Refs/references.bib",
  "localrefs.bib"])
# Gen .bcf file
out = check_output(["make4ht", "-u", "bib.tex"])
# Gen .bbl file
out = check_output(["biber", "bib.bcf"])
# Compile
out = check_output(["make4ht", "-u", "bib.tex"])
# Insert space between journal and number
with rewrite("bib.html") as lines:
  for i, line in enumerate(lines):
    lines[i] = re.sub(r"</span>(\d)", r"</span> \1", line)
# Convert html->rst
out = check_output(["pandoc", "-o", "bib.rst", "bib.html"])
# RST post-processing
with rewrite("bib.rst") as lines:
  for i, line in enumerate(lines):
    # Remove "` <bib.html>`__\ " crap
    line = line.replace(r"` <bib.html>`__\ ", "")
    # Convert opening and closing "
    line = line.replace(r"“", '"')
    line = line.replace(r"”", '"')
    # Convert   .. [ref]   to   [ref]_
    line = re.sub(r"^\s*(\[\w+\])", r".. \1",line)
    # Write
    lines[i] = line
#---------------------------------
