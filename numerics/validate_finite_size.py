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
CLASSICAL_FIRST = FIRST + RUNS
CLASSICAL_RUNS = 100_000
CLASSICAL_FILE = RESULTS / "validation_classical_100k.csv.gz"
SIZE = 1000
FIELDS = ("replicate", "seed", "n", "root_degree", "root_A", "root_Z")
SIZES = (1000, 10_000, 100_000, 1_000_000)


def raw_file(code):
    return RESULTS / f"validation_root_{code:02d}.csv.gz"


def simulate(binary, code, first, count, destination):
    """Use the original sampler and retain a disjoint block of replicate indices."""
    command = [str(binary.resolve()), str(code), str(first + count), str(SIZE)]
    with tempfile.TemporaryDirectory(dir=RESULTS) as directory:
        temporary = Path(directory) / "records.gz"
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as child:
            with temporary.open("wb") as file, gzip.GzipFile(
                filename="", mode="wb", fileobj=file, mtime=0
            ) as archive, io.TextIOWrapper(archive, newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                for row in csv.DictReader(child.stdout):
                    if int(row["replicate"]) >= first:
                        writer.writerow(row)
            if child.wait():
                raise RuntimeError(f"sampler failed at alpha={code / 100:g}")
        read_root_rows(temporary, code, first, count)
        temporary.replace(destination)


def classical_clock(n):
    return math.exp(sum(math.log1p(1 / (2 * (m - 1))) for m in range(2, n)))


def read_root_rows(path, code, first, count):
    exact_clock = classical_clock(SIZE) if code == 0 else None
    values = []
    with gzip.open(path, "rt", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream), start=first):
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
            if not math.isclose(value, ratio, rel_tol=2e-11):
                raise ValueError(f"inconsistent root measurement in {path}")
            values.append(value)
    if len(values) != count:
        raise ValueError(f"incorrect replicate count in {path}")
    return np.array(values)


def root_summary(code, z):
    exact_se = ""
    if code == 0:
        clock = classical_clock(SIZE)
        # E[D(D+1)] = 2*(n-1) and E[D] = A in the classical tree.
        variance = 2 * (SIZE - 1) / clock**2 - 1 / clock - 1
        exact_se = math.sqrt(variance / len(z))
    return (code / 100, SIZE, len(z), z.mean(), standard_error(z), exact_se)


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
            simulate(args.simulate, code, FIRST, RUNS, raw_file(code))
        z = read_root_rows(raw_file(code), code, FIRST, RUNS)
        roots.append(root_summary(code, z))
        fits.append(intercepts(code))
    header = ("alpha", "n", "runs", "mean_Z", "se_Z", "analytic_se_alpha_zero")
    write_rows(RESULTS / "validation_root_summary.csv", header, roots)
    if args.simulate is not None:
        simulate(args.simulate, 0, CLASSICAL_FIRST, CLASSICAL_RUNS, CLASSICAL_FILE)
    follow_up = read_root_rows(CLASSICAL_FILE, 0, CLASSICAL_FIRST, CLASSICAL_RUNS)
    write_rows(RESULTS / "validation_classical_summary.csv", header, [root_summary(0, follow_up)])
    write_root_panel(roots)
    write_rows(RESULTS / "validation_intercepts.csv",
               ("alpha", "runs", "four_size", "four_se", "three_size", "three_se",
                "three_minus_four", "paired_se"), fits)


if __name__ == "__main__":
    main()
