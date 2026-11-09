"""Check the mean of the root martingale and the stability of extrapolations."""
import argparse
import csv
import gzip
import io
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from observables import pathwise_coefficients, standard_error, write_rows


RESULTS = Path(__file__).resolve().parent / "results"
CODES = (0, 25, 50, 75, 90)
FIRST = 64  # indices 0 through 63 were available for the million-vertex runs
RUNS = 10_000
SIZE = 1000
FIELDS = ("replicate", "seed", "n", "root_degree", "root_A", "root_Z")
SIZES = (1000, 10_000, 100_000, 1_000_000)


def raw_file(code):
    return RESULTS / f"validation_root_{code:02d}.csv.gz"


def simulate(binary, code):
    """Use the original sampler and retain a disjoint block of replicate indices."""
    command = [str(binary.resolve()), str(code), str(FIRST + RUNS), str(SIZE)]
    with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
        temporary = Path(directory) / "records.gz"
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as child:
            with temporary.open("wb") as file, gzip.GzipFile(
                filename="", mode="wb", fileobj=file, mtime=0
            ) as archive, io.TextIOWrapper(archive, newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                for row in csv.DictReader(child.stdout):
                    if int(row["replicate"]) >= FIRST:
                        writer.writerow(row)
            if child.wait():
                raise RuntimeError(f"sampler failed at alpha={code / 100:g}")
        read_root_rows(temporary, code)
        temporary.replace(raw_file(code))


def classical_clock(n):
    return math.exp(sum(math.log1p(1 / (2 * (m - 1))) for m in range(2, n)))


def read_root_rows(path, code):
    with gzip.open(path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != RUNS:
        raise ValueError(f"incorrect replicate count in {path}")
    exact_clock = classical_clock(SIZE) if code == 0 else None
    z = []
    for index, row in enumerate(rows, start=FIRST):
        expected_seed = 20261108 + 1000003 * code + 1009 * index
        if (int(row["replicate"]), int(row["seed"]), int(row["n"])) != (
            index, expected_seed, SIZE
        ):
            raise ValueError(f"incorrect seed or checkpoint in {path}")
        value = float(row["root_Z"])
        clock = float(row["root_A"])
        if exact_clock is not None and not math.isclose(clock, exact_clock, rel_tol=2e-11):
            raise ValueError(f"incorrect classical root clock in {path}")
        ratio = int(row["root_degree"]) / clock
        if not np.isclose(value, ratio, rtol=2e-11, atol=0):
            raise ValueError(f"inconsistent root measurement in {path}")
        z.append(value)
    return np.array(z)


def intercepts(code):
    """Use the same independent paths for the four-size and three-size fits."""
    observations = np.genfromtxt(RESULTS / f"tree_{code:02d}.csv", names=True, delimiter=",")
    four = pathwise_coefficients(observations, code / 100, SIZES)[0]
    three = pathwise_coefficients(observations, code / 100, SIZES[1:])[0]
    change = three - four
    return (code / 100, len(four), four.mean(), standard_error(four),
            three.mean(), standard_error(three), change.mean(), standard_error(change))


def write_root_panel(rows):
    table = RESULTS.parent.parent / "tables" / "validation_rows.tex"
    table.parent.mkdir(exist_ok=True)
    with table.open("w") as out:
        out.write(r"\newcommand{\shortrootrows}{%" + "\n")
        for label, field, digits in ((r"\(\alpha\)", 0, 2), (r"Mean \(Z_{10^3,1}\)", 3, 4),
                                     ("Monte Carlo s.e.", 4, 4)):
            values = [f"{row[field]:.{digits}f}" for row in rows]
            out.write(" & ".join([label, *values]) + r" \\" + "\n")
        out.write("}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", type=Path, metavar="BINARY",
                        help="regenerate short-tree records before analysing them")
    args = parser.parse_args()
    roots, fits = [], []
    for code in CODES:
        if args.simulate is not None:
            simulate(args.simulate, code)
        z = read_root_rows(raw_file(code), code)
        exact_se = ""
        if code == 0:
            clock = classical_clock(SIZE)
            # E[D(D+1)] = 2*(n-1) and E[D] = A in the classical tree.
            variance = 2 * (SIZE - 1) / clock**2 - 1 / clock - 1
            exact_se = math.sqrt(variance / len(z))
        roots.append((code / 100, SIZE, len(z), z.mean(), standard_error(z), exact_se))
        fits.append(intercepts(code))
    write_rows(RESULTS / "validation_root_summary.csv",
               ("alpha", "n", "runs", "mean_Z", "se_Z", "analytic_se_alpha_zero"), roots)
    write_root_panel(roots)
    write_rows(RESULTS / "validation_intercepts.csv",
               ("alpha", "runs", "four_size", "four_se", "three_size", "three_se",
                "three_minus_four", "paired_se"), fits)


if __name__ == "__main__":
    main()
