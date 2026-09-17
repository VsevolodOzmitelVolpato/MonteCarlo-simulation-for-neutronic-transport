import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import os

PROJECT_DIR = r"C:\Users\sevin\OneDrive\Desktop\Uni\CppProgetti\Fermilab 2.0\main2 con scelta"
EXE = os.path.join(PROJECT_DIR, "main_2.exe")
MODE = "analog"
MATERIAL = "lead"
MAX_COLLISIONS = 100000
R1, R2, H = 100, 115, 500
N_BINS = 20

N_VALUES = [1000, 10000, 100000, 1000000, 10000000]

results = []
for N in N_VALUES:
    stdin_text = f"{MODE}\n{MATERIAL}\n{N}\n{MAX_COLLISIONS}\n{R1}\n{R2}\n{H}\n{N_BINS}\n"
    result = subprocess.run([EXE], input=stdin_text, capture_output=True, text=True, timeout=180, cwd=PROJECT_DIR)
    m = re.search(r"relative error \(outer\): ([\d.eE+-]+)", result.stdout)
    if not m:
        print(f"N={N}: parsing failed"); continue
    results.append((N, float(m.group(1))))
    print(f"N={N:>10}  relative error = {results[-1][1]:.6f}")

Ns = np.array([r[0] for r in results], dtype=float)
errs = np.array([r[1] for r in results])
slope, intercept = np.polyfit(np.log10(Ns), np.log10(errs), 1)
ref = errs[0] * (Ns / Ns[0])**(-0.5)

plt.loglog(Ns, errs, 'o-', label='relative error misured')
plt.loglog(Ns, ref, '--', color='gray', label='theoretical slope -1/2')
plt.xlabel("N particles"); plt.ylabel("relative error (outer)")
plt.title(f"Monte Carlo convergence — misured slope: {slope:.3f}")
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.show()