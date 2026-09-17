import subprocess
import re
import matplotlib.pyplot as plt
import os

PROJECT_DIR = r"C:\Users\sevin\OneDrive\Desktop\Uni\CppProgetti\Fermilab 2.0\main2 con scelta"
EXE = os.path.join(PROJECT_DIR, "main_2.exe")

MODE = "analog"
MATERIAL = "pure_absorber_test"
N_PARTICLES = 200000
MAX_COLLISIONS = 2000000
R1 = 10
THICKNESSES = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
HEIGHT = 500
N_BINS = 10

thicknesses_done, transmitted, transmitted_err, theory = [], [], [], []

for t in THICKNESSES:
    R2 = R1 + t
    stdin_text = f"{MODE}\n{MATERIAL}\n{N_PARTICLES}\n{MAX_COLLISIONS}\n{R1}\n{R2}\n{HEIGHT}\n{N_BINS}\n"
    try:
        result = subprocess.run([EXE], input=stdin_text, capture_output=True, text=True, cwd=PROJECT_DIR, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"t={t}: TIMEOUT")
        continue

    m_trans = re.search(r"outer escape\s*=\s*([\d.eE+-]+)\s*\+/-\s*([\d.eE+-]+)", result.stdout)
    m_theory = re.search(r"pure absorber\)\s*=\s*([\d.eE+-]+)", result.stdout)
    if not m_trans:
        print(f"t={t}: parsing failed")
        print("STDOUT:", repr(result.stdout))
        continue

    thicknesses_done.append(t)
    transmitted.append(float(m_trans.group(1)))
    transmitted_err.append(float(m_trans.group(2)))
    theory.append(float(m_theory.group(1)) if m_theory else None)

plt.errorbar(thicknesses_done, transmitted, yerr=transmitted_err, fmt='o', label='Monte Carlo', capsize=3)
if all(v is not None for v in theory):
    plt.plot(thicknesses_done, theory, '-', label='theory: exp(-Sigma_a * thickness)')
plt.xlabel("shell thickness R2-R1 (cm)")
plt.ylabel("outer escape proportion")
plt.legend()
plt.title("Transmission vs shell thickness")
plt.grid()
plt.show()