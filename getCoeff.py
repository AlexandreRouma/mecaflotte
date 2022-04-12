import numpy as np
import defs

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
    if type_cent == defs.NODE_TYPE_LIMIT:
        j = np.array([[num_cent]])
        a = np.array([[1]])
        b = cl_cent

    # If the node is an internal node, add the values for a laplacian computation
    # Note: bondery values are garanteed so there is only one formula
    elif type_cent == defs.NODE_TYPE_INTERNAL:
        j = np.array([[num_left], [num_right], [num_down], [num_up], [num_cent]])
        a = np.array([[1], [1], [1], [1], [-4]])
        b = 0

    # Otherwise, we add nothing to the system
    else:
        j = np.array([])
        a = np.array([])
        b = 0

    return j, a, b