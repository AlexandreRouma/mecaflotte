import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import defs
import time

from sameShape2D import sameShape2D
from laplace import laplace
from velocity import velocity

start = time.time()

# Configuration
CL_PATH     = "CL/3-cl.txt"
DOM_PATH    = "CL/3-dom.txt"
NUM_PATH    = "CL/3-num.txt"
DX          = 0.01
DY          = 0.01

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

# Generate dens matrix of values for ease of manipulation
phi = laplace(clMtx, domMtx, numMtx)

# Calculate velocity at each node
u, v = velocity(domMtx, phi, DX, DY)

end = time.time()

print(end - start)

# Display output
# plt.subplot(2, 2, 1)
# X = np.arange(0, NUM_COLUMNS)
# Y = np.arange(0, NUM_ROWS)
# AX, AY = np.meshgrid(X, Y)
# plt.streamplot(AX, AY, u, v)

# plt.subplot(2, 2, 2)
# plt.imshow(phi, cmap='turbo')
# plt.colorbar()

# plt.subplot(2, 2, 3)
plt.imshow(v, cmap='turbo')
plt.colorbar()

# plt.subplot(2, 2, 4)
# plt.imshow(v, cmap='turbo')
# plt.colorbar()

plt.show()