import numpy as np
import scipy
import defs
from getCoeff import getCoeff

# => Generate the laplacian system <=
# returns: A (system matrix), b (system second term), p (term position in matrix)
def buildSystem(clMtx, domMtx, numMtx):
    # Determin system size
    rows = clMtx.shape[0]
    columns = clMtx.shape[1]
    SYS_SIZE = 0
    for i in range(0, rows):
        for j in range(0, columns):
            if numMtx[i][j] > SYS_SIZE:
                SYS_SIZE = numMtx[i][j]
    
    # Allocate system members
    A = scipy.sparse.csc_matrix((SYS_SIZE, SYS_SIZE))
    b = scipy.sparse.csc_matrix((SYS_SIZE, 1))
    p = np.ndarray((SYS_SIZE, 2))

    # Fill out coefficients using the getCoeff function
    for i in range(0, rows):
        for j in range(0, columns):
            # If this is an external node, it's useless to add it to the system
            type_cent = domMtx[i][j]
            if type_cent == defs.NODE_TYPE_EXTERNAL:
                continue

            # Get number and optionally limit value of the center node
            cl_cent = clMtx[i][j]
            num_cent = numMtx[i][j]

            # Save position
            p[num_cent-1] = [i,j]

            # Get node number for the sides
            num_left = 0
            num_right = 0
            num_down = 0
            num_up = 0

            # If not on first line, get id of the node at the top
            if i > 0:
                num_up = numMtx[i - 1][j]
            
            # If not on the last line, get id of node at the bottom
            if i < rows-1:
                num_down = numMtx[i + 1][j]

            # If not on the first column, get id of node on the left
            if j > 0:
                num_left = numMtx[i][j - 1]

            # If not on the last column, get id of node on the right
            if j < columns-1:
                num_right = numMtx[i][j + 1]

            # Compute system additions
            vec_j, vec_a, val_b = getCoeff(num_left, num_right, num_down, num_up, num_cent, type_cent, cl_cent)

            # Add to the system
            # Note, we do -1 since node indices are 1 based, but matrix indicides are 0 based
            b[(num_cent-1, 0)] = val_b
            for k in range(0, len(vec_j)):
                A[(num_cent-1, vec_j[k][0]-1)] = vec_a[k][0]
            
    # Return generate system
    return A, b, p