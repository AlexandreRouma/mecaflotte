import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import defs

from sameShape2D import sameShape2D
from buildSystem import buildSystem
from plotNodes import plotNodes
from velocity import velocity

# Configuration
CL_PATH     = "3-cl.txt"
DOM_PATH    = "CL/3-dom.txt"
NUM_PATH    = "CL/3-num.txt"
DX          = 0.5
DY          = 0.5

# Load simulation data
clMtx = np.loadtxt(CL_PATH)
domMtx = np.loadtxt(DOM_PATH, dtype=int)
numMtx = np.loadtxt(NUM_PATH, dtype=int)

NUM_ROWS   = domMtx.shape[0]
NUM_COLUMNS    = domMtx.shape[1]

# Sanity check
if not sameShape2D(domMtx, numMtx) or not sameShape2D(domMtx, clMtx):
    print("ERROR: Input matrices are not of the same shape!")
    exit(-1)

# Solve for flow function
A, b, p = buildSystem(clMtx, domMtx, numMtx)
sol = scipy.sparse.linalg.spsolve(A,b)

# Generate dens matrix of values for ease of manipulation
flowMtx = plotNodes(sol, p, NUM_ROWS, NUM_COLUMNS)

# Calculate velocity at each node
u, v = velocity(flowMtx, domMtx, NUM_ROWS, NUM_COLUMNS)

# Display output
X = np.arange(0, NUM_COLUMNS)
Y = np.arange(0, NUM_ROWS)
AX, AY = np.meshgrid(X, Y)
plt.streamplot(AX, AY, u, v)
# plt.subplot(2, 1, 1)
# plt.imshow(u, cmap='turbo')
# plt.colorbar()
# plt.subplot(2, 1, 2)
# plt.imshow(np.abs(v), cmap='turbo')
# plt.colorbar()
plt.show()