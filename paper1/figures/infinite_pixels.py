import numpy as np
import matplotlib.pyplot as plt
import os

# Number of terms in output vector
K = 300

# Number of terms in input vector
N = 100

# Rank of the operator
R = 25

# Stability term
eps = 1e-12

# Random design matrix of shape (K, N) and rank R
np.random.seed(0)
A = np.random.randn(K, R) @ np.random.randn(R, N)

# Random input vector
y = np.random.randn(N)

# Output vector
f = A @ y

# Least squares solution
yls = np.linalg.lstsq(A, f, rcond=1e-12)[0]

# Now solve the least squares problem in a different basis
# We'll vary the dimension of this basis, `M`, and compute
# the fractional difference between the solutions in the
# two bases
M = np.array(np.logspace(2, 5, 50), dtype=int)
error1 = np.zeros(len(M))
error2 = np.zeros(len(M))
for m in range(len(M)):

    # Random change of basis matrix
    B = np.random.randn(M[m], N)

    # Pseudoinverse of B
    BInv = np.linalg.solve(B.T @ B + eps * np.eye(B.shape[1]), B.T)

    # Input vector in the B basis
    p = B @ y

    # Design matrix in the B basis
    A_ = A @ BInv

    # Check that in fact `p` gives us the same output `f`
    assert np.allclose(f, A_ @ p)

    # Least squares solution in the B basis
    pls = np.linalg.lstsq(A_, f, rcond=1e-12)[0]

    # Difference between the solution in the original basis
    error1[m] = np.mean((yls - BInv @ pls) ** 2) / np.mean(yls ** 2)

    # Difference between the solution in the B Basis
    error2[m] = np.mean((B @ yls - pls) ** 2) / np.mean(pls ** 2)

# Plot the errors
fig, ax = plt.subplots(1, figsize=(7, 7))
ax.plot(M, error1, label="Difference in original basis")
ax.plot(M, error2, label="Difference in new basis")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Dimension of new basis", fontsize=18)
ax.set_ylabel("Difference in preimage [fractional]", fontsize=18)
ax.legend(loc="upper right")

# We're done
fig.savefig(
    os.path.abspath(__file__).replace(".py", ".pdf"), bbox_inches="tight",
)
