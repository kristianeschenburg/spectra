import numpy as np
from spectra import adjacency as adj

from scipy import sparse

def normal(X, F, A=None):

    """
    Compute normals at each vertex.

    Parameters:
    - - - - -
    X: float, array
        vertex coordinates (m, 3)
    F: int, array
        triangles of mesh
    A: int, sparse array
        adjacency matrix of surface mesh
    
    Returns:
    - - - -
    N: float, array
        normal vectors at each vertex in mesh
    """

    n = X.shape[0]
    eps = 1e-6

    # generate adjacency matrix
    if not A:
        A = adj.unweightedadjacency(F)
    
    # compute vertex degree
    D = np.asarray(A.sum(1)).squeeze()

    # compute normals of each face
    Nf = ncrossp( X[F[:, 1], :]-X[F[:, 0], :], 
                  X[F[:, 2], :]-X[F[:, 0], :])

    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])

    d0 = np.concatenate([Nf[:, 0], Nf[:, 0], Nf[:, 0]])
    d1 = np.concatenate([Nf[:, 1], Nf[:, 1], Nf[:, 1]])
    d2 = np.concatenate([Nf[:, 2], Nf[:, 2], Nf[:, 2]])

    N = np.zeros((n, 3))
    N[:, 0] = sparse.csr_matrix((d0, (rows, cols)), shape=(n, n)).diagonal()
    N[:, 1] = sparse.csr_matrix((d1, (rows, cols)), shape=(n, n)).diagonal()
    N[:, 2] = sparse.csr_matrix((d2, (rows, cols)), shape=(n, n)).diagonal()
    N = N / D[:, None]

    dnorm = np.sqrt((N**2).sum(1))
    dnorm[dnorm < eps] = 1
    N = N / dnorm[:, None]

    return N


def ncrossp(x, y):

    """
    Compute cross product.

    Parameters:
    - - - - -
    x, y: float, array 
        3-dimensional (m, 3)
    """

    assert x.shape[1] == 3
    assert y.shape[1] == 3

    eps = 1e-6
    
    xnorm = np.sqrt((x**2).sum(1))
    xnorm[xnorm < eps] = 1
    x = x/xnorm[:, None]

    ynorm = np.sqrt((y**2).sum(1))
    ynorm[ynorm < eps] = 1
    y = y/ynorm[:, None]

    z = np.zeros((x.shape))
    z[:, 0] = x[:, 1]*y[:, 2] - x[:, 2]*y[:, 1]
    z[:, 1] = x[:, 2]*y[:, 0] - x[:, 0]*y[:, 2]
    z[:, 2] = x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0]

    znorm = np.sqrt((z**2).sum(1))
    znorm[znorm < eps] = 1
    z = z / znorm[:, None]

    return z