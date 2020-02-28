import numpy as np
from scipy import sparse

def degree(W):

    """
    Compute the weighted degree of each vertex in a surface mesh.

    Parameters:
    - - - - -
    W: float, sparse array
        weighted adjacency matrix of surface mesh

    Returns:
    - - - -
    D: float, array
        degree of each vertex
    Dinv: float, array
        inverse degree of each vertex
    """

    n = W.shape[0]
    D = sparse.spdiags(W.sum(1).squeeze(), diags=0, m=n, n=n)
    Dinv = sparse.spdiags(1/W.sum(1).squeeze(), diags=0, m=n, n=n)

    return [D, Dinv]

def spectrum(L, k=6):

    """
    Compute the smallest k eigenvalues and corresponding eigenvectors
    of the graph laplacian.

    Parameters:
    - - - - - 
    L: float, array
        sparse laplacian matrix
    k: int
        number of eigenvectors / values to compute

    Returns:
    - - - -
    E: float, array
        eigenvectors
    Lambda: float, array
        eigenvalues
    """

    [Lambda, E] = sparse.linalg.eigs(L, k=k, which='SM')
    Lambda = np.real(Lambda)
    E = np.real(E)

    # ensure that eigenvalues and vectors are sorted in ascending order
    idx = np.argsort(Lambda)
    Lambda = Lambda[idx]
    E = E[:, idx]

    # scale eigenvectors by inverse sqare root of eigenvales
    E[:,1:] = np.dot(E[:, 1:], np.diag(Lambda[1:]**(-0.5)))

    signf = 1-2*(E[0,:]<0)
    E = E*signf[None, :]

    return [E[:, 1:], Lambda[1:]]

def glaplacian(W, G=None):

    """
    Compute the general laplacian of the surface mesh.

    Parameters:
    - - - - -
    W: float, array
        weight adjacency matrix

    Returns:
    - - - -
    L: float, sparse array
        surface mesh Laplacian matrix
    """

    [D, Dinv] = degree(W)
    n = D.shape[0]

    if not G:
        G = D
        Ginv = Dinv
    
    L = np.dot(Ginv, (D - W))

    return L

def wlaplacian(L, T):

    """
    Weight laplacian matrix using surface features.

    Parameters:
    - - - - -
    L: float, array
        sparse laplacian matrix
    T: float, array
        scalar surface map (sulcal depth, cortical thickness, myelin density)

    Returns:
    - - - -
    L: float, sparse array
        surface mesh Laplacian matrix, weighted by surface scalar map
    """

    n = L.shape[0]
    a = 1
    G = sparse.spdiags(np.exp(T*a) - np.exp(T*a).min() + 1e-2, diags=0, m=n, n=n)
    L = np.dot(L, G)

    return L