import defs
import numpy as np
from deriv import deriv

# => Calculate velocity at each node using the flow matrix <=
# flowMtx   Flow matrix
# domMtx    Domain matrix
# u:        Horizontal component of velocity
# v:        Vertical component of velocity
# returns:  Pressure matrix
def pressure(domMtx, u, v, C, rho, g):
    # Allocate pressure matrix
    rows = domMtx.shape[0]
    columns = domMtx.shape[1]
    p = np.zeros((rows, columns))
    p[:] = np.nan

    # Calculate the pressure over the entire domain
    rho_g = rho * g
    two_g = 2.0 * g
    for i in range(0, rows):
        for j in range(0, columns):
            U2 = u[i,j]**2 + v[i,j]**2
            p[i,j] = rho_g * (C - U2/two_g)

    return p