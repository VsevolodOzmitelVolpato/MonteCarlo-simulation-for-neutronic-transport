import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(r"C:\Users\sevin\OneDrive\Desktop\Uni\CppProgetti\Fermilab 2.0\mu_lab_samples.txt", skiprows=1)

plt.hist(data, bins=50, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='mu = 0')
plt.xlabel("mu_L")
plt.ylabel("counter")
plt.title("mu_L distribution (A=1)")
plt.legend()
plt.show()

print("sample mean:", data.mean(), " (theoretical exact value for A=1: 2/3 =", 2/3, ")")

