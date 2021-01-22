import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.linalg import svd
import starry
import os

lmax = 16
nsamples = 100
np.random.seed(0)

# Compute the rank as a function of sph harm degree
# We'll do this for various inclinations and take
# the median.
map = starry.Map(lmax, lazy=False)
theta = np.linspace(0, 360, 1000)
R = np.empty((nsamples, lmax))
for k in range(nsamples):
    map.inc = 180 / np.pi * np.arccos(np.random.uniform(0, 1))
    A = map.design_matrix(theta=theta)
    R[k] = [
        np.linalg.matrix_rank(A[:, : (l + 1) ** 2]) for l in range(1, lmax + 1)
    ]
R = np.median(R, axis=0)

# Plot
fig, ax = plt.subplots(1, figsize=(6, 6))
l = np.arange(1, lmax + 1)
ax.plot(l, 2 * l + 1, "C0-", lw=1.5, label="number of light curve signals")
ax.plot(l, (l + 1) ** 2, "C1-", lw=1.5, label=r"number of surface modes")
ax.plot(l, R, "C0o", ms=5)
ax.legend(loc="upper left", fontsize=12)
ax.set_xlabel(r"spherical harmonic degree")
ax.set_ylabel(r"number")
ax.set_xticks([1, 5, 10, 15])

ax.annotate(
    "nullity",
    xy=(1.07, 0.55),
    xycoords="axes fraction",
    xytext=(0, 0),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=12,
    clip_on=False,
)
ax.annotate(
    "",
    xy=(1.07, 0.96),
    xycoords="axes fraction",
    xytext=(1.07, 0.585),
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=12,
    clip_on=False,
    arrowprops=dict(fc="k", width=0.2, headwidth=5, headlength=5),
)
ax.annotate(
    "",
    xy=(1.07, 0.15),
    xycoords="axes fraction",
    xytext=(1.07, 0.52),
    textcoords="axes fraction",
    ha="left",
    va="center",
    fontsize=12,
    clip_on=False,
    arrowprops=dict(fc="k", width=0.2, headwidth=5, headlength=5),
)

ax.annotate(
    "rank",
    xy=(1.07, 0.09),
    xycoords="axes fraction",
    xytext=(0, 0),
    textcoords="offset points",
    ha="center",
    va="center",
    fontsize=12,
    clip_on=False,
)
ax.annotate(
    "",
    xy=(1.07, 0.14),
    xycoords="axes fraction",
    xytext=(1.07, 0.11),
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=12,
    clip_on=False,
    arrowprops=dict(fc="k", width=0.2, headwidth=5, headlength=5),
)
ax.annotate(
    "",
    xy=(1.07, 0.04),
    xycoords="axes fraction",
    xytext=(1.07, 0.07),
    textcoords="axes fraction",
    ha="left",
    va="center",
    fontsize=12,
    clip_on=False,
    arrowprops=dict(fc="k", width=0.2, headwidth=5, headlength=5),
)


# Save
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight"
)
