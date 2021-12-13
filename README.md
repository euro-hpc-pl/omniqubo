## OMNIQUBO

[![codecov](https://codecov.io/gh/euro-hpc-pl/omniqubo/branch/master/graph/badge.svg?token=ysrSwHsXO8)](https://codecov.io/gh/euro-hpc-pl/omniqubo)
![quality_checks](https://github.com/euro-hpc-pl/omniqubo/actions/workflows/quality_checks.yml/badge.svg)
![tests](https://github.com/euro-hpc-pl/omniqubo/actions/workflows/tests.yml/badge.svg)

Simple, modular tool for transforming models into QUBO and related programs.
Currently, Quadratic Integer Programs with equality constraints written in docplex
can be transformed into BinaryQuadraticModel from dimod, and the results can be
interpreted into original form.