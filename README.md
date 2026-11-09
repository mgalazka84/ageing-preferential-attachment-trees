# Degree distributions and vertex growth in ageing preferential attachment trees

Marek Gałązka and Hanna Wdowicka

This repository contains the code and simulation results for the numerical experiments in the paper. At each step a new vertex attaches to one older vertex. The probability of selecting a vertex is proportional to its degree times its age raised to the power `-alpha`.

## Reproduce the figures and tables

From the repository root, run the following commands with Python, NumPy, SciPy and Matplotlib installed.

```sh
cd numerics
python check.py
python analyse.py
python validate_finite_size.py
```

The main analysis reads the five tree_XX.csv files in [numerics/results](numerics/results). Each row records one tree at one size. The 160 independent trees were observed at four sizes, from 1,000 to 1,000,000 vertices, giving 640 observations in total. The four rows for a given seed belong to the same tree.

A separate check uses 10,000 new trees of size 1,000 for each of the five ageing exponents. The [root-martingale means](numerics/results/validation_root_summary.csv) can be reproduced from the five compressed per-tree CSVs. The [paired extrapolations](numerics/results/validation_intercepts.csv) compare three versus four observation sizes on the same million-vertex trees.

The [reproduction instructions](numerics/README.txt) give the compiler flags, exact seeds, commands for regrowing the trees, a separate numerical check of the root normalisation, and definitions of the data columns.

- [Simulator](numerics/simulate.cpp) and [root normalisation](numerics/root_clock.hpp)
- [Mass-equation solver and analysis](numerics/analyse.py)
- [Degree and tail calculations](numerics/observables.py)
- [Vector figures](figures) and [generated table entries](tables)
