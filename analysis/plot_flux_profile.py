import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\sevin\OneDrive\Desktop\Uni\CppProgetti\Fermilab 2.0\bin_fluc_unc.csv")

plt.errorbar(df["bin center"], df["flux"], yerr=df["uncertainty"], fmt='o-', capsize=3, markersize=3)
plt.xlabel("radial distance from axis  (cm)")
plt.ylabel("flux (per source particle)")
plt.title("spatial flux profile")
plt.grid()
plt.show()

