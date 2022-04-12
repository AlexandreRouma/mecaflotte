# => Check if two matrices are of the same shape and size <=
# m1:       First matrix
# m2:       Second matrix
# returns:  True if matrices are of the same shape and size
def sameShape2D(m1, m2):
    return m1.shape[0] == m2.shape[0] and m1.shape[1] == m2.shape[1]