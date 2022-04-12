import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

# Enums
NULL                = 0
NODE_TYPE_EXTERNAL  = 0
NODE_TYPE_INTERNAL  = 1
NODE_TYPE_LIMIT     = 2

# => Get the coefficients to add to the laplacian system <=
# num_left:     Index of the node to the left
# num_right:    Index of the node to the right
# num_down:     Index of the node below
# num_up:       Index of the node above
# type_cent:    node type
# cl_cent:      Limit condition value (optional)
# returns:      j (row of the added values), a (values added to the matrix), b (values added to the second term)
def getCoeff(num_left, num_right, num_down, num_up, num_cent, type_cent, cl_cent):
    # If the node is a limit node, the b value is simply the node value
    if type_cent == NODE_TYPE_LIMIT:
        j = np.array([[num_cent]])
        a = np.array([[1]])
        b = cl_cent

    # If the node is an internal node, add the values for a laplacian computation
    # Note: bondery values are garanteed so there is only one formula
    elif type_cent == NODE_TYPE_INTERNAL:
        j = np.array([[num_left], [num_right], [num_down], [num_up], [num_cent]])
        a = np.array([[1], [1], [1], [1], [-4]])
        b = 0

    # Otherwise, we add nothing to the system
    else:
        j = np.array([])
        a = np.array([])
        b = 0

    return j, a, b

# => Derive value <=
# f_left:       function value on the left
# f_c           function value in the center
# f_right:      function value on the right
# type_left:    node type on the left
# type_c:       node type in the center
# type_right:   node type on the right
# h:            node step size
# returns:      left to right derivative
def deriv(f_left,f_c,f_right,type_left,type_c,type_right,h):
    # Do not derive out-of-bounds nodes
    if type_c == NODE_TYPE_EXTERNAL:
        v = 0

    # If the left side is out-of-bounds, do right asymetric derivative
    elif type_left == NODE_TYPE_EXTERNAL:
        v = (f_right - f_c) / h

    # If the right side is out-of-bounds, do left asymetric derivative
    elif type_right == NODE_TYPE_EXTERNAL:
        v = (f_c - f_left) / h

    # Otherwise, do a symetric derivative
    else:
        v  = (f_right - f_left) / (2.0 * h)

    return v

# => Calculate velocity at each node using the flow matrix <=
# flowMtx   Flow matrix
# domMtx    Domain matrix
# dx:       X space increment
# dy:       Y space increment
# returns:  u: Matrix of horizontal velocities, v: Matrix of verical velocities
def velocity(domMtx, flowMtx, dx, dy):
    # Allocate horizontal and vertical speed matrices
    rows = domMtx.shape[0]
    columns = domMtx.shape[1]
    u = np.zeros((rows, columns))
    v = np.zeros((rows, columns))

    # Iterate over all elements of the flow matrix
    for l in range(0, rows):
        for c in range(0, columns):
            # If a node is external, skip it
            if domMtx[l][c] == defs.NODE_TYPE_EXTERNAL:
                u[l][c] = np.NaN
                v[l][c] = np.NaN
                continue

            # Default types
            ltype = defs.NODE_TYPE_EXTERNAL
            utype = defs.NODE_TYPE_EXTERNAL
            ctype = domMtx[l][c]
            dtype = defs.NODE_TYPE_EXTERNAL
            rtype = defs.NODE_TYPE_EXTERNAL

            # Default values
            lval = 0.0
            uval = 0.0
            cval = flowMtx[l][c]
            dval = 0.0
            rval = 0.0

            # Set types and values using domain matrix
            if l > 0:
                utype = domMtx[l-1][c]
                uval = flowMtx[l-1][c]
            if  l < rows - 1:
                dtype = domMtx[l+1][c]
                dval = flowMtx[l+1][c]
            if c > 0:
                ltype = domMtx[l][c-1]
                lval = flowMtx[l][c-1]
            if  c < columns - 1:
                rtype = domMtx[l][c+1]
                rval = flowMtx[l][c+1]

            # Calculate derivatives
            v[l][c] = -deriv(uval, cval, dval, utype, ctype, dtype, dx)
            u[l][c] = deriv(lval, cval, rval, ltype, ctype, rtype, dy)
    
    return u, v

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
            if type_cent == NODE_TYPE_EXTERNAL:
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

def submit(which):
    dom = np.loadtxt(which+ '-dom.txt', dtype = int)
    num = np.loadtxt(which+ '-num.txt', dtype = int)
    cl = np.loadtxt(which+ '-cl.txt', dtype = float) # Les conditions limites sont imposées et ne doivent donc pas être déterminées
    if which == '1':
        dx = 0.5
    else:
        dx = 0.01
    
    psi = laplace(cl, dom, num)

    u, v = velocity(dom, psi, dx, dx)

    return psi,u,v
