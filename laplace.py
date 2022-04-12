import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import defs
from getCoeff import getCoeff

# => Generate the laplacian system <=
# clMtx:    Derichlet conditions matrix
# domMtx:   Domain matrix
# numMtx:   Node number matrix
# returns:  Matrix containing the value of the flow function at each node
def laplace(clMtx, domMtx, numMtx):
    # Determin system size
    rows = clMtx.shape[0]
    columns = clMtx.shape[1]
    sysSize = 0
    for i in range(0, rows):
        for j in range(0, columns):
            if numMtx[i][j] > sysSize:
                sysSize = numMtx[i][j]
    
    # Allocate system members
    pos = np.ndarray((sysSize, 2))

    # Lists
    all_i = []
    all_j = []
    all_a = []
    all_bi = []
    all_b = []

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
            pos[num_cent-1] = [i,j]

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

            # Get system values
            vec_j, vec_a, val_b = getCoeff(num_left, num_right, num_down, num_up, num_cent, type_cent, cl_cent)
            
            # Append positions and A values
            all_i.extend(np.full(len(vec_j), num_cent - 1))
            all_j.extend(vec_j.transpose()[0] - 1)
            all_a.extend(vec_a.transpose()[0])
            
            # Append b values
            all_bi.append(num_cent - 1)
            all_b.append(val_b)
            
    A = scipy.sparse.csc_matrix((all_a,(all_i, all_j)))
    b = scipy.sparse.csc_matrix((all_b,(all_bi, np.full(len(all_bi), 0))))

    # Solve system
    sol = scipy.sparse.linalg.spsolve(A,b)

    # Allocate dens array
    psi = np.empty((rows, columns))
    psi[:] = np.nan

    # Fill with solution values
    for i in range(0, sysSize):
        psi[int(pos[i][0])][int(pos[i][1])] = sol[i]

    return psi