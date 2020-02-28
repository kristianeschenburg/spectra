import numpy as np
from scipy import sparse

from spectra import normal as norm

def unweightedadjacency(F):

    """
    Compute adjacency matrix of surface mesh

    Parameters:
    - - - - -
    F: int, array
        mesh triangles

    Returns:
    - - - -
    A: int, sparse array
        adjacency matrix of surface mesh
    """

    n = F.max()+1

    rows = np.concatenate([F[:, 0], F[:, 0], 
                            F[:, 1], F[:, 1], 
                            F[:, 2], F[:, 2]])

    cols = np.concatenate([F[:, 1], F[:, 2], 
                            F[:, 0], F[:, 2], 
                            F[:, 0], F[:, 1]])

    combos = np.column_stack([rows, cols])

    [_, idx] = np.unique(combos, axis=0, return_index=True)
    A = sparse.csr_matrix((np.ones(len(idx)), (combos[idx, 0], combos[idx, 1])), shape=(n, n))

    return A

def weightedadjacency(X, F):
    
    n,p = V.shape
    
    weights = np.sqrt(np.concatenate([((X[F[:, 0], :] - X[F[:, 1], :])**2).sum(1),
                                      ((X[F[:, 0], :] - X[F[:, 2], :])**2).sum(1),
                                      ((X[F[:, 1], :] - X[F[:, 0], :])**2).sum(1),
                                      ((X[F[:, 1], :] - X[F[:, 2], :])**2).sum(1),
                                      ((X[F[:, 2], :] - X[F[:, 0], :])**2).sum(1),
                                      ((X[F[:, 2], :] - X[F[:, 1], :])**2).sum(1),]))
    
    eps = 1e-6
    
    rows = np.concatenate([F[:, 0], F[:, 0],
                         F[:, 1], F[:, 1],
                         F[:, 2], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2],
                          F[:, 0], F[:, 2],
                          F[:, 0], F[:, 1]])
    
    combos = np.column_stack([rows, cols])
    [rc, idx] = np.unique(combos, axis=0, return_index=True)
    weights = weights[idx]
    
    W = sparse.csr_matrix((weights, (rc[:, 0], rc[:, 1])), shape=(n, n))
    W = (W + W.transpose())/2
    
    return W


def weightedadjacencynormal(X, F):

    """
	Compute weighted adjacency matrix.

	Parameters:
	- - - - - -
	X: float, array
		vertex coordinates
	F: int, array
		mesh triangles

	Returns:
	- - - -
	W: float, sparse array
		weighted adjacency matrix
	"""

    eps = 1e-6
    N = norm.normal(X, F)
    n = X.shape[0]

    # compute weights for all links (euclidean distance)
    wdist = np.sqrt(np.concatenate([((X[F[:, 0], :]-X[F[:, 1], :])**2).sum(1),
                                ((X[F[:, 0], :]-X[F[:, 2], :])**2).sum(1),
                                ((X[F[:, 1], :]-X[F[:, 0], :])**2).sum(1),
                                ((X[F[:, 1], :]-X[F[:, 2], :])**2).sum(1),
                                ((X[F[:, 2], :]-X[F[:, 0], :])**2).sum(1),
                                ((X[F[:, 2], :]-X[F[:, 1], :])**2).sum(1)]))

    # compute weights for all links (euclidean distance)
    wnormal = np.sqrt(np.concatenate([((N[F[:, 0], :]-N[F[:, 1], :])**2).sum(1),
                                ((N[F[:, 0], :]-N[F[:, 2], :])**2).sum(1),
                                ((N[F[:, 1], :]-N[F[:, 0], :])**2).sum(1),
                                ((N[F[:, 1], :]-N[F[:, 2], :])**2).sum(1),
                                ((N[F[:, 2], :]-N[F[:, 0], :])**2).sum(1),
                                ((N[F[:, 2], :]-N[F[:, 1], :])**2).sum(1)]))
    
    wdist   /= wdist.mean()
    wnormal /= wnormal.mean()
    weights = (wdist + wnormal + eps)**(-1)

    rows = np.concatenate([F[:, 0], F[:, 0], 
                            F[:, 1], F[:, 1], 
                            F[:, 2], F[:, 2]])

    cols = np.concatenate([F[:, 1], F[:, 2], 
                            F[:, 0], F[:, 2], 
                            F[:, 0], F[:, 1]])

    combos = np.column_stack([rows, cols])

    [rc, idx] = np.unique(combos, axis=0, return_index=True)
    weights = weights[idx]

    W = sparse.csr_matrix((weights, (rc[:, 0], rc[:, 1])), shape=(n, n))

    return W