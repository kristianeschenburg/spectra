import numpy as np

def filter(V, F, inds):

    """
    Filter vertices and faces by list of indices.

    Parameters:
    - - - - -
    V: float, array
        x,y,z coordinates of the surface mesh
    F: int, array
        triangles of the surface mesh
    inds: int, list
        list of indices to keep
    """

    inds.sort()
    indmap = dict(zip(inds, np.arange(len(inds))))

    V = V[inds, :]

    gface = []
    for face in F:
        check = np.zeros(3)
        
        for j in np.arange(3):
            check[j] = (face[j] in inds)
        
        if check.sum() == 3:
            nface = [indmap[f] for f in face]
            gface.append(nface)
    
    gface = np.row_stack(gface)

    return [V, gface]
        