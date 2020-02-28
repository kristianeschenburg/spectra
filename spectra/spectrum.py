from scipy.sparse import csr, linalg
from spectra import adjacency, graphs, normal

class Spectrum(object):

    """
    Class to compute the eigenvectors of a graph Laplacian,
    specifically as it refers to mesh based data.  Inputs 
    are the vertices and faces of the mesh.  Outputs are the
    eigenvectors of the graph laplacian.

    Parameters:
    - - - - -
    V: float, array
        x,y,z coordinates of the surface mesh
    F: int, array
        triangles of the surface mesh
    wan: bool
        Compute adjacency matrix weighted by 
        mesh normal vectors
    """

    def __init__(self, V, F, adj= None, n_components=6, wadj=True, wlap=False, features=None):

        self.vertices = V
        self.faces = F
        self.n_components = n_components
        self.wadj = wadj

        if wlap:
            if features:
                try:
                    features.shape == V[:, 0].shape
                except:
                    wlap = False
                    features = None
        else:
            weight_lap = None

        self.adj = adj
        self.wlap = wlap
        self.features = features

    def fit(self):

        V = self.vertices
        F = self.faces

        if isinstance(self.adj, csr.csr_matrix):
            W = self.adj
        else:

            if self.wadj:
                W = adjacency.weightedadjacencynormal(V, F)
            else:
                W = adjacency.unweightedadjacency(F)

        L = graphs.glaplacian(W, G=None)
        if self.wlap:
            L = graphs.wlaplacian(L, self.features)
        
        [evecs, evals] = graphs.spectrum(L, self.n_components+1)
        
        self.evecs = evecs
        self.evals = evals
    
