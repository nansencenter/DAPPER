---
title: 'DAPPER: Data Assimilation with Python: a Package for Experimental Research'
tags:
  - data-assimilation
  - state-estimation
  - bayesian-filter
  - kalman-filtering
  - enkf
  - particle-filter
  - chaos
  - bayesian-methods
authors:
  - name: Patrick N. Raanes
    orcid: 0000-0001-5073-6786
    corresponding: true
    affiliation: "1, 2"
  - name: Yumeng Chen
    orcid: 0000-0002-2319-6937
    affiliation: "3"
  - name: Colin Grudzien
    orcid: 0000-0002-3084-3178
    affiliation: "4, 5"
affiliations:
 - name: NORCE, Bergen, Norway
   index: 1
 - name: NERSC, Bergen, Norway
   index: 2
 - name: Department of Meteorology and NCEO, University of Reading, Reading, UK
   index: 3
 - name: CW3E - Scripps Institution of Oceanography, University of California, San Diego, USA
   index: 4
 - name: Department of Mathematics and Statistics, University of Nevada, Reno, USA
   index: 5
date: 10 January 2023
bibliography: paper.bib
---

# Summary

Data assimilation (DA) is the science of optimally combining sparse data and
computational models, both of which are typically large and varied.
Based on classical statistics, signal processing, and control systems theory,
and increasingly, machine learning, DA was primarily developed in weather,
climate, and oceanographic forecasting but is presently used all across the
geosciences and beyond.
DAPPER is a set of templates for benchmarking the performance of DA methods.

![Screenshot from "liveplotting" in DAPPER](dapper-L63.png)

# Statement of need

DAPPER provides experimental support and guidance for new developments in DA
by facilitating numerical investigations through a variety of
typical test cases and statistics. It (a) reproduces numerical benchmarks
results reported in the literature and (b) facilitates comparative studies,
thus promoting the (a) reliability and (b) relevance of the results.
DAPPER is (c) open source, written in Python, and (d) focuses on readability;
this promotes the (c) reproduction and (d) dissemination of the underlying
science, and makes it easy to adapt and extend.

The README contains a comprehensive list of related projects,
a list of 5 publications (to date) that used DAPPER,
and a long list of results published in the literature that have been reproduced with DAPPER,
of which we mention [@asch2016data; @bocquet2014iterative; @anderson2010non].

# Outlook

The intention is for DAPPER to continue
benchmarking and illustrating the latest methods in DA and beyond.
For example, at the time of writing,
there are 26 open ticket in the repository's issue tracker.
Most have been opened by the principle investigator,
and are tagged with *enhancement*.
These are not necessarily minor, and have mainly been filed
under the `v2` milestone.

# Acknowledgements

We acknowledge support from Alberto Carrassi and Laurent Bertino of NERSC
in the genesis of this project, continued financial backing from NORCE,
and contributions from Maxime Tondeur, and Remy Dubois,
