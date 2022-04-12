import numpy as np

# => Calculate force applied to a object using the pressures at the nodes <=
# p:        vector containing the pressure at each node
# x:        vector containing the x positions of the nodes
# y:        vector containing the y positions of the nodes
# returns:  circulation around the nodes
def force(p,x,y):
    # Sanity check
    vcount = len(p)
    if vcount != len(x) or vcount != len(y):
        raise Exception("The number of pressures, x positions and y positions must match")

    # Note: we use 3 here because the first node is duplicated at the end
    if vcount < 3:
        raise Exception("At least two nodes are required to calculate the circulation")

    # Verify that curve is closed
    if x[0] != x[vcount-1] or y[0] != y[vcount-1]:
        raise Exception("Curve must be closed")

    # Prepare work values
    f = np.array([0.0, 0.0])
    zvec = np.array([0.0, 0.0, 1.0])
    lastPress = p[0]
    lastPos = np.array([x[0], y[0]])

    # Perform trapezoidal integration
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