import numpy as np
from scipy.linalg import svd


def test_svd():
    # Number of terms in output vector
    K = 100

    # Number of terms in input vector
    N = 50

    # Rank of the operator
    R = 25

    # Random design matrix of shape (K, N) and rank R
    np.random.seed(0)
    A = np.random.randn(K, R) @ np.random.randn(R, N)

    # SVD decomposition of A
    U, S, VT = svd(A)

    # Preimage operator
    VTx = VT[:R]
    PRE = VTx.T @ VTx

    # Null space operator
    VTo = VT[R:]
    NULL = VTo.T @ VTo

    # Random input vector
    y = np.random.randn(N)

    # Preimage component of input vector
    yx = PRE @ y

    # Null space component of input vector
    yo = NULL @ y

    # Output vector
    f = 1 + A @ y

    # Least squares solution
    yls = np.linalg.lstsq(A, f - 1, rcond=1e-12)[0]

    # Check that the least squares solution IS the preimage
    assert np.allclose(yx, yls)
