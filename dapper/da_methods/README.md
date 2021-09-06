See the README section on
[DA Methods](https://github.com/nansencenter/DAPPER#DA-Methods)
for an overview of the methods included with DAPPER.

## Defining your own method

Follow the example of one of the methods within one of the
sub-directories/packages.
The simplest example is perhaps
`dapper.da_methods.ensemble.EnKF`.

## General advice for programming/debugging scientific experiments

- Start with something simple.
  This helps make sure the basics of the experiment are reasonable.
  For example, start with

      - a pre-existing example,
      - something you are able to reproduce,
      - a small/simple model.

        - Set the observation error to be small.
        - Observe everything.
        - Don't include model error and/or noise to begin with.

- Additionally, test a simple/baseline method to begin with.
  When including an ensemble method, start with using a large ensemble,
  and introduce localisation later.

- Take incremental steps towards your ultimate experiment setup.
  Validate each incremental setup with prints/plots.
  If results change, make sure you understand why.

- Use short experiment duration.
  You probably don't need statistical significance while debugging.
