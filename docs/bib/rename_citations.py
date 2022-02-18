#!/usr/bin/env python

"""Search DAPPER for (e.g.) `bib.bocquet2011ensemble` and change to `bib.bocquet2011ensemble`."""

import os
from dapper.dpr_config import rc
from patlib.std import rewrite, sub_run


renaming = dict(
    Boc14="bocquet2014iterative",
    And10="anderson2010non",
    Liu01="liu2001theoretical",
    Raa19a="raanes2019adaptive",
    Lei11="lei2011moment",
    Kar07="karspeck2007experimental",
    Boc13="bocquet2013joint",
    Dou09="doucet2009tutorial",
    Wik07="wikle2007bayesian",
    Raa15="raanes2015rts",
    Hun07="hunt2007efficient",
    Che03="chen2003bayesian",
    Sak08a="sakov2008deterministic",
    Sak08b="sakov2008implications",
    Boc15="bocquet2015expanding",
    Boc12="bocquet2012combining",
    van09="van2009particle",
    Zup05="zupanski2005maximum",
    Sak12="sakov2012iterative",
    Hot15="hoteit2015mitigating",
    Boc16="bocquet2016localization",
    Raa19b="raanes2019revising",
    Wil16="wiljes2016second",
    Raa16b="raanes2016thesis",
    Dou01="doucet2001sequential",
    Töd15="todter2015second",
    Eve09="evensen2009ensemble",
    Boc11="bocquet2011ensemble",
)


def rename_citation(file_list):
    "Extract references in use."
    for f in file_list:
        with rewrite(f) as lines:
            for i, ln in enumerate(lines):
                for old, new in renaming.items():
                    old = f"`bib.{old}`"
                    new = f"`bib.{new}`"
                    ln = ln.replace(old, new)
                lines[i] = ln


# Get list of files to search for references
os.chdir(rc.dirs.DAPPER)
gitfiles = sub_run(["git", "ls-tree", "-r", "--name-only", "HEAD"])
gitfiles = [f for f in gitfiles.split("\n") if f.endswith(".py")]

rename_citation(gitfiles)
