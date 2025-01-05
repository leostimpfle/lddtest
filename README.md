# lddtest

Perform equivalence testing for regression discontinuity designs (RDD) in Python.

This package implements three methodologies:

1. The classic manipulation test proposed by [McCrary (2008)](https://www.sciencedirect.com/science/article/pii/S0304407607001133): tested against `DCdensity` implemented in the R package [`rdd`](https://github.com/ddimmery/rdd/blob/master/DCdensity.R)
2. The equivalence testing framework proposed by [Hartman (2021)](https://www.cambridge.org/core/journals/political-analysis/article/equivalence-testing-for-regression-discontinuity-designs/43F77CDC6337A63AE0A5E6DC3EE01A41): tested against the R functions [`rdd_equivalence`](https://github.com/ekhartman/rdd_equivalence)
3. The logarithmic density discontinuity (LDD) test proposed by [Fitzgerald (2024)](https://www.econstor.eu/handle/10419/300277): tested against the R package [`eqtesting`](https://github.com/jack-fitzgerald/eqtesting)