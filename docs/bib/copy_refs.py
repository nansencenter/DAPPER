#!/usr/bin/env python

"""
Search DAPPER for `bib.someRef` and copy the entry from `references.bib`.

Copies into `refs.bib` which can then be processed with `make_bib.py`.

Note: We don't want to merge these scripts
because `refs.bib` can also be edited manually,
and `references.bib` is my own centralized database.
"""

import re
from pathlib import Path

# from subprocess import run
# from textwrap import dedent
import os
from dapper.dpr_config import rc
from dapper.tools.colors import coloring
from colorama import Fore as CF
from patlib.std import sub_run


def parse_bib(bibfile):
    """Parse .bib file into dict."""
    bibfile = open(bibfile).readlines()
    refs = {}
    name = None
    for line in bibfile:

        if name:
            # Append to current reference block
            refs[name].append(line)

            if line.lstrip().startswith("}"):
                # Quit reference block
                refs[name] = "".join(refs[name])
                name = None

        # Detect start of reference block
        elif line.lstrip().startswith("@"):
            try:
                # Start block
                name = line[line.index("{") + 1 :].strip(" \n,")
            except IndexError:
                continue
            refs[name] = [line]
    return refs


def parse_citations(file_list):
    "Extract references in use."
    PATTERN = r"`bib\.\w{0,30}`"
    citations = []
    for f in file_list:
        f = open(f).read()
        cc = re.findall(PATTERN, f)
        citations.extend(cc)

    # Unique
    citations = set(citations)

    # Rm known false positives
    citations.discard("`bib.bocquet2011ensemble`")
    citations.discard("`bib.someRef`")
    citations.discard("`bib.py`")
    citations.discard("`bib.`")

    # Strip `bib.`
    citations = {c.strip("`")[4:] for c in citations}

    return citations


# Get list of files to search for references
os.chdir(rc.dirs.DAPPER)
gitfiles = sub_run(["git", "ls-tree", "-r", "--name-only", "HEAD"])
gitfiles = [f for f in gitfiles.split("\n") if f.endswith(".py")]

citations = parse_citations(gitfiles)

references_bib = Path("~/P/Refs/references.bib").expanduser()
refs_bib = rc.dirs.DAPPER / "docs" / "bib" / "refs.bib"
references = parse_bib(references_bib)
refs = parse_bib(refs_bib)

# print(*gitfiles, sep="\n")
# print(citations)
# for k in references:
#     print(k)
#     print(*references[k])

with open(refs_bib, "a") as out:
    for c in citations:

        if c in refs:
            with coloring(CF.BLUE):
                print("Already there:", c)
            pass

        elif c in references:
            with coloring(CF.GREEN):
                print("Copying:      ", c)
            out.write("\n")
            out.write(references[c])

        else:
            with coloring(CF.RED):
                print("Not found:    ", c)
