import numpy as np

# => Generate matrix from value/position pairs <=
# nval:     Node values
# npos      Node positions
# returns:  Dense matrix built from the pairs
def plotNodes(nval, npos, rows, columns):
    # Sanity check
    if len(nval) != len(npos):
        raise Exception("Mismatch in size between value and position arrays")

    # Allocate new matrix
    mat = np.zeros((rows, columns))

    # Fill with values at given positions
    for i in range(0, len(npos)):
        mat[int(npos[i][0])][int(npos[i][1])] = nval[i]

    return mat