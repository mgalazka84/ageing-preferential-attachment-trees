Numerical supplement
Degree distributions and vertex growth in ageing preferential attachment trees
Marek Gałązka and Hanna Wdowicka

The numerical files in this directory reproduce the three numerical figures,
tables, and finite-tree diagnostics. Run the commands below from this directory.
The five tree_XX.csv files contain the observations from the 160 million-vertex trees.
The five validation_root_XX.csv.gz files contain 50,000 additional short trees.
validation_classical_100k.csv.gz contains a separate 100,000 short trees at zero ageing.
Both analyses can be run directly on the supplied records.

Analysis and checks
-------------------
  python check.py
  python analyse.py
  python validate_finite_size.py
  c++ -O3 -std=c++17 -fopenmp-simd check_clock.cpp -o check_clock
  ./check_clock

analyse.py uses NumPy, SciPy and Matplotlib. It reads results/tree_XX.csv,
writes summary CSVs back to results/, prepares the figures in ../figures/
and exports the two original LaTeX table bodies to ../tables/. It checks the mass equation
against a separate ODE integration and checks classical probabilities and
tails against exact formulas. observables.py contains the extrapolation,
root, complementary distribution and table routines. check.py independently
enumerates all 5913 recursive trees of sizes 2 through 8 at each of the five
ageing exponents and verifies the rejection probabilities. check_clock.cpp
compares the positive exponential scheme and accumulated log product with
direct sums at all steps on star, path and mixed attachment histories.

Reproduce the simulations
-------------------------
  c++ -O3 -std=c++17 -fopenmp-simd simulate.cpp -o simulate
  ./simulate 0  24 1000000 > results/tree_00.csv
  ./simulate 25 24 1000000 > results/tree_25.csv
  ./simulate 50 64 1000000 > results/tree_50.csv
  ./simulate 75 24 1000000 > results/tree_75.csv
  ./simulate 90 24 1000000 > results/tree_90.csv
  python analyse.py

The optional GCC flag -fopenmp-simd enables vectorisation of the kernel
updates; the published runs also enabled -march=native. The sampler uses
a 64-bit Mersenne Twister (std::mt19937_64). Seeds are
20261108 + 1000003 * alpha_percent + 1009 * replicate, starting at replicate 0.
Sizes 1000, 10000, 100000 and 1000000 are observations of the SAME tree.
The experiments used GCC 13.3.0, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0
and Matplotlib 3.10.8.

Meaning of the data
-------------------
tree_XX.csv in results/ gives one row per seed and size. T_scaled is the exact
direct attachment-weight sum divided by n^(1-alpha), evaluated at checkpoints
in floating point. root_degree is the degree of vertex 1, and proposals is the
cumulative rejection-proposal count. cohort1--cohort20 give the mean degrees
in the respective relative-birth-time intervals ((i-1)/20,i/20]. degree1--
degree16 are per-tree degree frequencies. root_A is the predictable product
A_(n,1), root_Z is root_degree/root_A, and max_degree is the largest observed
degree. Each aboveK column is an INTEGER COUNT of degrees strictly greater
than K. Divide by n to obtain a fraction, or pool equal-size replicates.

beta.csv: scalar roots, growth and frequency exponents, and independent mass
residuals. tree_summary.csv: finite-tree normalisation estimates and standard
errors, median and quartiles of d_n(1)/n^beta, and leaf comparisons.
root_martingale.csv: means and standard errors of root_Z and leaf fractions.
finite_size_fits.csv: coefficients of a + c*n^(-(1-alpha)), plus standard
errors across independent four-size tree histories. degree_probabilities.csv:
the geometric-mixture probabilities for k=1,...,16 and their simulated values.
tails.csv: pooled exceedance counts, mean empirical survival fractions and
their between-tree standard errors, limiting survival fractions, and the
large-k approximations (including thresholds with zero observations).

Numerical root clock
--------------------
Power weights used for parent selection are precomputed directly. To measure
the product A_(n,1) on its realised path, root_clock.hpp approximates
j^(-alpha) by a positive 74-term sum c_l exp(-s_l*(j-1)) at alpha>0.
The nodes use the log-Laplace grid log s_l = l/2, l=-64,...,8, and a
constant term for the lower grid remainder. The approximation is compared
with the power at EVERY integer age up to the requested size before use.
At time n each state is the degree-weighted sum of one exponential. If v is
chosen, its update is exp(-s_l)*state + 1 + exp(-s_l*(n-v+1)).
Two lookup tables split each exponent's integer age into a quotient and
remainder modulo 1024. Exponential factors below exp(-48) are discarded.
The per-path product accumulates log1p(n^(-alpha)/T_n). At alpha=0 the
normaliser is exactly 2(n-1). Direct checkpoints and small-size tests are
documented in the manuscript.

Finite-size validation
----------------------
  python validate_finite_size.py
  # To regrow all short trees first, compile simulate.cpp as above, then run
  python validate_finite_size.py --simulate ./simulate

The study uses 10,000 trees of size 1,000 for each of the five ageing
exponents. The sampler and seed formula are the same as in the main study.
Replicate indices 64 through 10063 are retained, inclusive. Each seed occurs
on one validation path only and is distinct from all main-study seeds.
The compressed CSVs record the replicate, seed, size, root degree, root_A
and root_Z. The deterministic no-ageing clock is checked against the
product of 1 + 1/(2*(m-1)) from m=2 to n-1 for every classical replicate.

validation_root_summary.csv gives the count, mean and Monte Carlo standard
error (sample standard deviation divided by the square root of the count).
The exact standard error is also given at zero ageing. There, with D the
root degree and A its clock, E[D]=A and E[D*(D+1)]=2*(n-1), so Var(D/A)
equals 2*(n-1)/A**2 - 1/A - 1. The script reproduces the validation panel
of the normalisation table in ../tables/validation_rows.tex.

validation_intercepts.csv uses the original 160 trees, with the same paths
in both fits. It gives the intercept and its Monte Carlo standard error
using all four sizes, the same quantities using the three largest sizes,
and the paired difference (three minus four) with its standard error.
Each unweighted least-squares fit uses one tree's observed scaled weights.
These standard errors describe simulation variation conditional on the fit.

Independent repetition at zero ageing
-------------------------------------
The same validation command also analyses validation_classical_100k.csv.gz
and writes validation_classical_summary.csv. The repetition uses n=1,000,
alpha=0 and replicate indices 10064 through 110063, inclusive. It uses the
same sampler and file columns as the five original validation batches.
Including --simulate regrows all six batches. Earlier replicate indices
encountered while starting the sampler are discarded for this repetition.

For the tail-count comparison at alpha=0.50, multiply the mixture and
power_asymptotic columns of tails.csv by runs*n to obtain the two limiting
benchmarks for the pooled counts. The maximum degree of each main-study tree
is in tree_50.csv at its million-vertex checkpoint.
