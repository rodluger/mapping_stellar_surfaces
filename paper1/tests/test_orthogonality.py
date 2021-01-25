import numpy as np
from scipy.linalg import svd


def test_orthogonality():
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

    # The preimage component of V^T
    VTx = VT[:R]

    # The null space component of V^T
    VTo = VT[R:]

    # Check the orthogonality of VTx
    assert np.allclose(VTx @ VTx.T, np.eye(R))

    # Check the orthogonality of VTo
    assert np.allclose(VTo @ VTo.T, np.eye(N - R))
