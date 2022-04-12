import defs
import numpy as np
from deriv import deriv

# => Calculate velocity at each node using the flow matrix <=
# flowMtx   Flow matrix
# domMtx    Domain matrix
# rows:     Number of rows
# columns:  Number of columns
# returns:  u: Matrix of horizontal velocities, v: Matrix of verical velocities
def velocity(flowMtx, domMtx, rows, columns):
    # Allocate horizontal and vertical speed matrices
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
            u[l][c] = deriv(uval, cval, dval, utype, ctype, dtype, 1.0)
            v[l][c] = -deriv(lval, cval, rval, ltype, ctype, rtype, 1.0)
    
    return u, v