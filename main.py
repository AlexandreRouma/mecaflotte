from binascii import b2a_base64
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Enums
NULL                = 0
NODE_TYPE_EXTERNAL  = 0
NODE_TYPE_INTERNAL  = 1
NODE_TYPE_LIMIT     = 2

# Configuration
CL_PATH     = "2-cl.txt"
DOM_PATH    = "CL/2-dom.txt"
NUM_PATH    = "CL/2-num.txt"
DX          = 0.5
DY          = 0.5

# Load simulation data
clMtx = np.loadtxt(CL_PATH)
domMtx = np.loadtxt(DOM_PATH, dtype=int)
numMtx = np.loadtxt(NUM_PATH, dtype=int)

NUM_LINES   = domMtx.shape[0]
NUM_ROWS    = domMtx.shape[1]

# Sanity check
def sameShape2D(m1, m2):
    return m1.shape[0] == m2.shape[0] and m1.shape[1] == m2.shape[1]

if not sameShape2D(domMtx, numMtx) or not sameShape2D(domMtx, clMtx):
    print("ERROR: Input matrices are not of the same shape!")
    exit(-1)

# ======== Helper Functions ========

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

# => Generate the laplacian system <=
# returns: A (system matrix), b (system second term), p (term position in matrix)
def buildLaplacianSystem():
    # Determin system size
    SYS_SIZE = 0
    for i in range(0, NUM_LINES):
        for j in range(0, NUM_ROWS):
            if numMtx[i][j] > SYS_SIZE:
                SYS_SIZE = numMtx[i][j]
    
    # Allocate system members
    A = scipy.sparse.csc_matrix((SYS_SIZE, SYS_SIZE))
    x = scipy.sparse.csc_matrix((SYS_SIZE, 1))
    b = scipy.sparse.csc_matrix((SYS_SIZE, 1))
    p = np.ndarray((SYS_SIZE, 2))

    # Fill out coefficients using the getCoeff function
    for i in range(0, NUM_LINES):
        for j in range(0, NUM_ROWS):
            # If this is an external node, it's useless to add it to the system
            type_cent = domMtx[i][j]
            if type_cent == NODE_TYPE_EXTERNAL:
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
            if i < NUM_LINES-1:
                num_down = numMtx[i + 1][j]

            # If not on the first column, get id of node on the left
            if j > 0:
                num_left = numMtx[i][j - 1]

            # If not on the last column, get id of node on the right
            if j < NUM_ROWS-1:
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

# => Calculate circulation around nodes <=
# u:        vector containing the horizontal components of the speeds
# v:        vector containing the vertical components of the speeds
# x:        vector containing the x positions of the nodes
# y:        vector containing the y positions of the nodes
# returns:  circulation around the nodes
def circu(u,v,x,y):
    # Sanity check
    vcount = len(u)
    if vcount != len(v) or vcount != len(x) or vcount != len(y):
        print("ERROR: Cannot calculate circulations if all data for each vector isn't given")
        exit(-1)

    # Note: we use 3 here because the first node is duplicated at the end
    if vcount < 3:
        print("ERROR: At least two nodes are required to calculate the circulation")
        exit(-1)

    # Verify that curve is closed
    if x[0] != x[vcount-1] or y[0] != y[vcount-1]:
        print("ERROR: Curve must be closed")
        exit(-1)
    
    # Prepare work values
    c = 0.0
    lastVec = np.array([u[0], v[0]])
    lastPos = np.array([x[0], y[0]])
    
    # Perform trapezoidal integration for all the next nodes
    for i in range(1, vcount):
        # Get new node speed vector
        vec = np.array([u[i], v[i]])
        pos = np.array([x[i], y[i]])

        # Sanity check
        if pos[0] == lastPos[0] and pos[1] == lastPos[1]:
            print("ERROR: Two consecutive nodes are at the exact same position, cannot integrate...")
            exit(-1)

        # Calculate the trangant vector between the last and new node. Then calculate its length and normalize it
        tan = pos - lastPos
        dx = np.linalg.norm(tan)
        ntan = tan * (1.0 / dx)

        # Calculate each side of the trapeze
        a = np.dot(lastVec, ntan)
        b = np.dot(vec, ntan)

        # Calculate the trapeze integral and add it to the sum
        c += dx*(a + b) / 2.0
        
        # Save last values
        lastVec = vec
        lastPos = pos

    return c

# => Calculate force applied to a object using the pressures at the nodes <=
# p:        vector containing the pressure at each node
# x:        vector containing the x positions of the nodes
# y:        vector containing the y positions of the nodes
# returns:  circulation around the nodes
def force(p,x,y):
    # Sanity check
    vcount = len(p)
    if vcount != len(x) or vcount != len(y):
        print("ERROR: The number of pressures, x positions and y positions must match")
        exit(-1)

    # Note: we use 3 here because the first node is duplicated at the end
    if vcount < 3:
        print("ERROR: At least two nodes are required to calculate the circulation")
        exit(-1)

    # Verify that curve is closed
    if x[0] != x[vcount-1] or y[0] != y[vcount-1]:
        print("ERROR: Curve must be closed")
        exit(-1)

    # Prepare work values
    f = np.array([0.0, 0.0])
    zvec = np.array([0.0, 0.0, 1.0])
    lastPress = p[0]
    lastPos = np.array([x[0], y[0]])

    # Add up all the forces around the curve
    for i in range(1, vcount):
        # Get new node position and pressure
        press = p[i]
        pos = np.array([x[i], y[i]])

        # Average out the pressure over the source
        apress = (press + lastPress) / 2.0

        # Calculate normal to the curve
        tan = pos - lastPos
        vec3d = np.array([-tan[0], -tan[1], 0])
        norm = np.cross(vec3d, zvec)
        norm = norm * (1.0 / np.linalg.norm(norm))
        tlen = np.linalg.norm(tan)

        # Add force
        f += norm[0:2] * (apress * tlen)

        lastPress = press
        lastPos = pos

    return f[0], f[1]

# A, b, p = buildLaplacianSystem()

# x = scipy.sparse.linalg.spsolve(A,b)

# out = np.zeros((NUM_LINES, NUM_ROWS))
# for i in range(0, len(p)):
#     out[int(p[i][0])][int(p[i][1])] = x[i]

# print(out)

# plt.imshow(out)
# plt.show()