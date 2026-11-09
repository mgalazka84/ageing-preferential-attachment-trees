"""Replicate-level diagnostics and the complementary degree distribution."""
import csv
from math import exp, expm1, lgamma, log
from scipy.special import digamma
import matplotlib.pyplot as plt
import numpy as np


def write_rows(path, header, rows):
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(rows)


def standard_error(values):
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def pathwise_coefficients(raw, alpha, sizes):
    """Fit a + c*n**(alpha-1) to the selected checkpoints of each tree."""
    sizes = np.asarray(sizes)
    paths = []
    for replicate in np.unique(raw["replicate"]):
        records = raw[(raw["replicate"] == replicate) & np.isin(raw["n"], sizes)]
        records = np.sort(records, order="n")
        if not np.array_equal(records["n"], sizes):
            raise ValueError("missing or duplicate checkpoint in a pathwise fit")
        paths.append(records["T_scaled"])
    design = np.column_stack((np.ones(len(sizes)), sizes ** (alpha - 1.0)))
    return np.linalg.lstsq(design, np.array(paths).T, rcond=None)[0]


def export(results, figures, exponent, integrated_weight, integrate_y, degree_pmf):
    alphas = (0., .25, .5, .75, .9)
    fit_rows, root_rows, tail_rows, pmf_rows = [], [], [], []
    tail_data = {}
    for alpha in alphas:
        raw = np.genfromtxt(results / f"tree_{round(100*alpha):02d}.csv",
                            names=True, delimiter=",")
        sizes = np.unique(raw["n"]).astype(int)
        runs = np.unique(raw["replicate"]).astype(int)
        beta = exponent(alpha)
        coefficients = pathwise_coefficients(raw, alpha, sizes)
        fit_rows.append((alpha, len(runs), coefficients[0].mean(),
                         standard_error(coefficients[0]), coefficients[1].mean(),
                         standard_error(coefficients[1]), 1/beta))
        for n in sizes:
            at_size = raw[raw["n"] == n]
            z = at_size["root_Z"]
            leaves = at_size["degree1"]
            root_rows.append((alpha, n, len(z), z.mean(), standard_error(z),
                              leaves.mean(), standard_error(leaves), degree_pmf(alpha, 1)))
        final = raw[raw["n"] == sizes[-1]]
        thresholds = np.array([int(col[5:]) for col in raw.dtype.names
                               if col.startswith("above")], dtype=int)
        c_alpha = -digamma(1-alpha)-np.euler_gamma
        log_constant = c_alpha + lgamma(1 + 1/beta)
        tail_points = []
        for k in thresholds:
            # Work in log coordinates and rescale the integral at a point
            # where its mass is appreciable. This keeps very small tails visible.
            def log_integrand(y):
                hazard = beta*integrated_weight(alpha, y)
                return -y + k*log(-expm1(-hazard)) if hazard > 0. else -np.inf
            scale = max(log_integrand(float(y)) for y in
                        np.linspace(.01, max(8., log(1+k)/beta + 2), 65))
            upper = max(72., -scale + 45.)
            theory = exp(scale)*integrate_y(lambda y: exp(log_integrand(y)-scale), 0., upper)
            power = exp(log_constant - log(k)/beta)
            counts = final[f"above{k}"]
            shares = counts / sizes[-1]
            tail_rows.append((alpha, int(sizes[-1]), len(final), k, int(counts.sum()),
                              shares.mean(), standard_error(shares), theory, power))
            tail_points.append((k, int(counts.sum()), shares.mean(), theory, power))
            if alpha == 0:
                analytic = 2./((k+1)*(k+2))
                assert abs(theory/analytic-1) < 2.e-10
        tail_data[alpha] = np.array(tail_points)
        for k in range(1,17):
            prob = degree_pmf(alpha,k)
            observed = final[f"degree{k}"]
            pmf_rows.append((alpha, k, prob, observed.mean(), standard_error(observed)))

    write_rows(results/"finite_size_fits.csv",
               ["alpha","runs","intercept","intercept_se","slope","slope_se","limit"], fit_rows)
    write_rows(results/"root_martingale.csv",
               ["alpha","n","runs","Z_mean","Z_se","leaf_mean","leaf_se","leaf_limit"], root_rows)
    write_rows(results/"tails.csv",
               ["alpha","n","runs","k","exceedances","empirical","empirical_se",
                "mixture","power_asymptotic"], tail_rows)
    write_rows(results/"degree_probabilities.csv",
               ["alpha","k","mixture","empirical","empirical_se"], pmf_rows)
    tail_figure(tail_data, figures/"degree-tails.pdf")


def tail_figure(data, filename):
    from matplotlib.ticker import LogLocator, NullFormatter
    fig, axes = plt.subplots(1,2,figsize=(5.20,2.63))
    for idx,(ax, alpha) in enumerate(zip(axes,(0.,.5))):
        values=data[alpha]
        k,count,obs,theory,power=values.T
        ax.loglog(k,theory,color="#17626c",linewidth=1.55,label="Geometric mixture")
        # The asymptote is a large-k comparison.
        region = k>=8
        ax.loglog(k[region],power[region],ls="--",color="#454555",
                  linewidth=1.12,label="Power-law asymptote")
        usable = count>=20
        ax.loglog(k[usable],obs[usable],linestyle="none",marker="o",ms=3.5,
                  mfc="none",mec="#ad542b",mew=.8,label="Simulated trees")
        ax.set_xlim(.85,1300); ax.set_ylim(1e-10,1.1)
        ax.set_xlabel("Degree threshold $k$")
        ax.set_title(f"({'ab'[idx]}) "+rf"$\alpha={alpha:g}$",loc="left",fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=8, width=.6, length=3)
        ax.grid(axis="y",color="#d5d8da",linewidth=.45)
        if idx==0:
            ax.set_ylabel("Fraction with degree above $k$")
            ax.legend(loc="lower left",frameon=False,fontsize=6.7,handlelength=1.5)
    fig.subplots_adjust(left=.117,right=.982,top=.88,bottom=.19,wspace=.35)
    temp = filename.with_suffix(".tmp")
    fig.savefig(temp,format="pdf")
    plt.close(fig)
    temp.replace(filename)

def export_table_rows(results, tables):
    """Write the two numeric table bodies used by the LaTeX source."""
    tables.mkdir(exist_ok=True)
    norm = np.genfromtxt(results/"tree_summary.csv", names=True, delimiter=",")
    root = np.genfromtxt(results/"root_martingale.csv", names=True, delimiter=",")
    fit = np.genfromtxt(results/"finite_size_fits.csv", names=True, delimiter=",")
    with (tables/"normalisation_rows.tex").open("w") as out:
        out.write(r"\newcommand{\normalisationrows}{%"+"\n")
        for alpha in np.unique(norm["alpha"]):
            d = np.sort(norm[norm["alpha"]==alpha],order="n")
            z = root[(root["alpha"]==alpha) & (root["n"]==max(d["n"]))][0]
            fields = [f"{alpha:.2f}",str(int(z["runs"]))]
            fields += [f"{v:.3f}" for v in d["T_mean"]]
            fields += [f'{d["T_limit"][-1]:.3f}',
                       f'{z["Z_mean"]:.3f} ({z["Z_se"]:.3f})']
            out.write(" & ".join(fields)+r" \\"+"\n")
        out.write("}\n")
    with (tables/"fit_rows.tex").open("w") as out:
        out.write(r"\newcommand{\fitrows}{%"+"\n")
        for d in fit[fit["alpha"]>0]:
            fields = [f'{d["alpha"]:.2f}', f'{d["intercept"]:.5f}',
                      f'{d["intercept_se"]:.5f}', f'{d["slope"]:.3f}',
                      f'{d["limit"]:.5f}']
            out.write(" & ".join(fields)+r" \\"+"\n")
        out.write("}\n")
