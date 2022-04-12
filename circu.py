import numpy as np

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
        raise Exception("Cannot calculate circulations if all data for each vector isn't given")

    # Note: we use 3 here because the first node is duplicated at the end
    if vcount < 3:
        raise Exception("At least two nodes are required to calculate the circulation")

    # Verify that curve is closed
    if x[0] != x[vcount-1] or y[0] != y[vcount-1]:
        raise Exception("Curve must be closed")
    
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
            raise Exception("Two consecutive nodes are at the exact same position, cannot integrate...")

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