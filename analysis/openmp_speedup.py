"""Measure the OpenMP strong-scaling speedup of the Monte Carlo transport code.

Runs the same fixed problem with 1, 2, 4, ... threads, repeats each run, and
reports median time, speedup and parallel efficiency. It also prints the physical
result so you can check that all thread counts agree within the statistical error
(they will not be bit-identical: each thread has its own random stream).

--control
    Also measures the hardware's own scaling limit. For every thread count t, t
    independent single-thread copies of the program are started at the same time (no
    shared memory, no shared heap) and
        limit(t) = t * T_omp(1) / T_slowest_copy(t)
    where T_omp(1) is the one-thread OpenMP time and T_slowest_copy(t) the time the
    slowest of the t copies needs while the others run. The slowest copy is the
    reference because the slowest thread also sets the time of an OpenMP loop (the
    loop ends at an implicit barrier). A laptop CPU lowers its clock when all cores are
    busy, and the two hyper-threads of a core share one set of execution units, so the
    limit is below t. OpenMP/limit = speedup / limit compares the OpenMP run with what
    the same CPU does for independent processes: 100% means nothing is lost to the code
    (shared memory, allocator, scheduling). The last column is the stricter version that
    uses the mean copy time (what a perfectly load-balanced loop could reach). For each
    t the OpenMP runs and the copy runs alternate, so both see the same machine state.
    The 1-copy row compares two runs of the same command: its distance from 100% is the
    noise of the whole measurement.

--compare OTHER_EXE
    A/B test of two executables (for example before and after a code change). For
    every thread count the runs alternate A,B / B,A / A,B ... so that clock, thermal
    and background-load drift affect both equally. Each repeat gives one paired ratio
    time(B)/time(A) from two runs made back to back. The table gives the median ratio,
    a distribution-free interval for the true median and the number of repeats in which
    B was faster. B/A < 1 means B is faster. The interval comes from order statistics:
    each paired ratio lies below the true median with probability 1/2, so the number
    of ratios below it is Binomial(n, 1/2) and [r(k), r(n+1-k)] (sorted ratios) covers
    it with probability 1 - 2*P(Bin(n, 1/2) <= k-1). If the whole interval is below
    1.00 the difference is resolved; if it contains 1.00 it is not. Do not judge by
    the min-max of the ratios: it widens as the number of repeats grows (for 15
    repeats it is a 99.99% interval). To measure the noise floor and any order bias of
    the method, compare an executable with a byte-for-byte copy of itself (A/A test):
    B/A must come out at 1.00 and the interval must contain it.

Usage (from the folder that contains materials.txt):
    python openmp_speedup.py path/to/your_executable
    python openmp_speedup.py path/to/your_executable --threads 1 2 4 8 --repeats 5
    python openmp_speedup.py path/to/your_executable --control --repeats 5
    python openmp_speedup.py path/to/exe_A --compare path/to/exe_B --threads 8 16 --repeats 15

The executable must be the interactive program (full_model / main_2) compiled with
-fopenmp. Inputs are fed in the same order the program prompts for them.
Note: every run overwrites bin_fluc_unc.csv in the current folder.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb

# Same case as the README: lead shell, implicit capture, 2e6 histories.
DEFAULT_INPUT = ["implicit", "lead", "2000000", "1000", "100", "115", "500", "20"]


def run_once(exe, n_threads, inputs):
    env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
    out = subprocess.run([exe], input="\n".join(inputs) + "\n", capture_output=True,
                         text=True, env=env, check=True).stdout
    time_s = float(re.search(r"time:\s*([0-9.eE+-]+)\s*s", out).group(1))
    m = re.search(r"outer escape\s*=\s*([0-9.eE+-]+)\s*\+/-\s*([0-9.eE+-]+)", out)
    return time_s, float(m.group(1)), float(m.group(2))


def run_copies(exe, n_copies, inputs):
    """Start n_copies single-thread copies at the same time; return each copy's loop time."""
    with ThreadPoolExecutor(max_workers=n_copies) as pool:
        futures = [pool.submit(run_once, exe, 1, inputs) for _ in range(n_copies)]
        return [f.result()[0] for f in futures]


def median_interval(values):
    """Distribution-free interval for the median: [x(k), x(n+1-k)] of the sorted values.

    k is the largest that keeps the coverage 1 - 2*P(Bin(n, 1/2) <= k-1) at or above 90%
    (below 5 values even the min-max falls short; the coverage is returned so it is shown).
    """
    x = sorted(values)
    n = len(x)
    dropped = 0
    for j in range(n // 2):
        if 1 - 2 * sum(comb(n, i) for i in range(j + 1)) / 2 ** n >= 0.90:
            dropped = j
        else:
            break
    coverage = 1 - 2 * sum(comb(n, i) for i in range(dropped + 1)) / 2 ** n
    return x[dropped], x[n - 1 - dropped], coverage


def scaling(a, threads):
    """Strong scaling, optionally with the hardware-limit control (see the docstring)."""
    base = None
    speedup = {}
    copies = {}
    print(f"{'threads':>7} {'median time [s]':>16} {'speedup':>8} {'efficiency':>11} {'outer escape':>22}")
    for t in threads:
        print(f"  ... measuring {t} thread(s)", file=sys.stderr, flush=True)
        omp_runs, slowest, average = [], [], []
        for _ in range(a.repeats):
            omp_runs.append(run_once(a.exe, t, a.inputs))
            if a.control:
                times = run_copies(a.exe, t, a.inputs)
                slowest.append(max(times))
                average.append(statistics.mean(times))
        med = statistics.median(r[0] for r in omp_runs)
        base = base or med
        speedup[t] = base / med
        mean, err = omp_runs[0][1], omp_runs[0][2]
        print(f"{t:>7} {med:>16.3f} {speedup[t]:>8.2f} {speedup[t] / t:>10.0%} {mean:>14.6f} +/- {err:.6f}")
        if a.control:
            copies[t] = (statistics.median(slowest), statistics.median(average))

    if a.control:
        print("\nHardware limit: t independent single-thread copies run at the same time")
        print(f"(limit = t * T_omp(1) / T_slowest_copy(t), with T_omp(1) = {base:.3f} s; the 1-copy row "
              f"should read 100%, its deviation is the measurement noise)")
        print(f"{'copies':>7} {'slowest copy [s]':>17} {'mean copy [s]':>14} {'limit':>8} "
              f"{'OpenMP/limit':>13} {'OpenMP/limit (mean copy)':>25}")
        for t in threads:
            slow, avg = copies[t]
            limit, limit_mean = t * base / slow, t * base / avg
            print(f"{t:>7} {slow:>17.3f} {avg:>14.3f} {limit:>8.2f} "
                  f"{speedup[t] / limit:>13.0%} {speedup[t] / limit_mean:>25.0%}")


def compare(a, threads):
    """A/B test of two executables with alternating runs (see the docstring)."""
    print(f"A = {a.exe}\nB = {a.compare}")
    print("runs alternate A,B / B,A so both see the same machine state; B/A < 1 means B is faster\n")
    print(f"{'threads':>7} {'A median [s]':>13} {'B median [s]':>13} {'B/A':>6} {'interval':>13} "
          f"{'B faster':>9} {'same outer escape':>18}")
    coverage = None
    for t in threads:
        print(f"  ... measuring {t} thread(s)", file=sys.stderr, flush=True)
        times_a, times_b, ratios, same = [], [], [], True
        for r in range(a.repeats):
            order = [("A", a.exe), ("B", a.compare)] if r % 2 == 0 else [("B", a.compare), ("A", a.exe)]
            res = {name: run_once(exe, t, a.inputs) for name, exe in order}
            times_a.append(res["A"][0])
            times_b.append(res["B"][0])
            ratios.append(res["B"][0] / res["A"][0])
            same = same and res["A"][1:] == res["B"][1:]
        lo, hi, coverage = median_interval(ratios)
        wins = sum(x < 1.0 for x in ratios)
        print(f"{t:>7} {statistics.median(times_a):>13.3f} {statistics.median(times_b):>13.3f} "
              f"{statistics.median(ratios):>6.3f} {lo:>6.3f}-{hi:<6.3f} "
              f"{wins:>5}/{len(ratios):<3} {'yes' if same else 'NO':>18}")
    print(f"\ninterval: distribution-free {coverage:.1%} interval for the median of the paired ratios "
          f"({a.repeats} repeats). Entirely below 1.00: B is faster. Contains 1.00: not resolved.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exe")
    p.add_argument("--threads", type=int, nargs="+", default=None)
    p.add_argument("--repeats", type=int, default=5)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--control", action="store_true",
                      help="also measure the hardware scaling limit with independent copies")
    mode.add_argument("--compare", metavar="OTHER_EXE",
                      help="A/B test: time this executable against OTHER_EXE with alternating runs")
    p.add_argument("--inputs", nargs="+", default=DEFAULT_INPUT)
    a = p.parse_args()

    n_cpu = os.cpu_count() or 1
    threads = sorted(set(a.threads or ({1, 2, 4, 8, n_cpu} & set(range(1, n_cpu + 1)))))
    if a.control and threads[0] != 1:
        threads = [1] + threads            # the one-thread time is the reference of the limit
    print(f"logical CPUs: {n_cpu}, thread counts: {threads}, repeats: {a.repeats}\n")

    if a.compare:
        compare(a, threads)
    else:
        scaling(a, threads)


if __name__ == "__main__":
    main()
