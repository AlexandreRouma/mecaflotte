import numpy as np
import matplotlib.pyplot as plt
import time
from pressure import pressure
from force import force
from circu import circu

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

def incRange(a, b):
    r = []
    if a < b:
        for i in range(a, b+1):
            r.append(i)
    else:
        for i in range(b, a+1):
            r.append(i)
        r.reverse()
    return r

def genFlowCL(cl, flowRate, dx, x0, x1, y0):
    width = np.abs(x1 - x0) * dx
    velocity = flowRate / width
    dphi = velocity * dx
    phi = 0.0
    for x in incRange(x0, x1):
        cl[y0,x] = phi
        phi += dphi
    return 0.0, phi - dphi

def genStaticHCL(cl, phi, x0, x1, y0):
    for x in incRange(x0, x1):
        cl[y0,x] = phi

def genStaticVCL(cl, phi, x0, y0, y1):
    for y in incRange(y0, y1):
        cl[y,x0] = phi

def genStaticRectCL(cl, phi, x0, y0, x1, y1):
    genStaticHCL(cl, phi, x0, x1, y0)
    genStaticVCL(cl, phi, x1, y0, y1)
    genStaticHCL(cl, phi, x1, x0, y1)
    genStaticVCL(cl, phi, x0, y1, y0)

GROUP_X = 5
GROUP_Y = 2
Q = (10.0*GROUP_X + 5.0*GROUP_Y) * 1e-3

# Clear limit conditions
clMtx = np.zeros((NUM_ROWS, NUM_COLUMNS))
clMtx[:] = np.nan

# Top
phi0, phi1 = genFlowCL(clMtx, Q, DX, 13, 38, 1)

# Bottom
phi0_o, phi1_o = genFlowCL(clMtx, Q, DX, 18, 33, 200)

# Left size
genStaticVCL(clMtx, phi0, 13, 1, 21)
genStaticHCL(clMtx, phi0, 13, 1, 21)
genStaticVCL(clMtx, phi0, 1, 21, 180)
genStaticHCL(clMtx, phi0, 1, 18, 180)
genStaticVCL(clMtx, phi0, 18, 180, 200)

# Right size
genStaticVCL(clMtx, phi1, 38, 1, 21)
genStaticHCL(clMtx, phi1, 38, 50, 21)
genStaticVCL(clMtx, phi1, 50, 21, 180)
genStaticHCL(clMtx, phi1, 50, 33, 180)
genStaticVCL(clMtx, phi1, 33, 180, 200)

genFlowCL(clMtx, Q, DX, 18, 33, 200)

# Obstruction
ASYMETRY = 0.5
genStaticRectCL(clMtx, ASYMETRY*phi0 + (1.0 - ASYMETRY)*phi1, 10, 35, 41, 107)

# Generate dens matrix of values for ease of manipulation
phi = laplace(clMtx, domMtx, numMtx)

# Calculate velocity
u, v = velocity(domMtx, phi, DX, DY)
U = np.sqrt(u**2 + v**2)

# Calculate pressure
p = pressure(domMtx, u, v, 0.0, 1000.0, 9.81)

# Build outline
def addHLine(x, y, x0, x1, y0):
    for i in incRange(x0, x1):
        x.append(i)
        y.append(y0)

def addVLine(x, y, x0, y0, y1):
    for i in incRange(y0, y1):
        x.append(x0)
        y.append(i)

def addRect(x, y, x0, y0, x1, y1):
    addHLine(x, y, x0, x1, y0)
    addVLine(x, y, x1, y0, y1)
    addHLine(x, y, x1, x0, y1)
    addVLine(x, y, x0, y1, y0)

vp = []
x = []
y = []

addRect(x, y, 10, 35, 41, 107)

for i in range(0, len(x)):
    vp.append(p[y[i],x[i]])

fu, fv = force(vp, x, y)

print("FY:", fu * DX, "FX:", fv * DY)

end = time.time()

print("Exec time: ", end - start)

START_POINTS = []
for i in incRange(13, 38):
    START_POINTS.append([1, i])

# , start_points=START_POINTS, density=10

# Display output
plt.subplot(3, 2, 1)
X = np.arange(0, NUM_COLUMNS)
Y = np.arange(0, NUM_ROWS)
AX, AY = np.meshgrid(X, Y)
plt.streamplot(AY.transpose(), AX.transpose(), u.transpose(), v.transpose(), color=U.transpose(), cmap='turbo', start_points=START_POINTS, density=10)
plt.colorbar()
plt.title("Stream Lines")

plt.subplot(3, 2, 2)
plt.imshow(phi.transpose(), cmap='turbo')
plt.colorbar()
plt.title("Flow Function")

plt.subplot(3, 2, 3)
plt.imshow(clMtx.transpose(), cmap='turbo')
plt.colorbar()
plt.title("Limit Conditions")

plt.subplot(3, 2, 5)
plt.imshow(U.transpose(), cmap='turbo')
plt.colorbar()
plt.title("Velocity")

plt.subplot(3, 2, 6)
plt.imshow(p.transpose(), cmap='turbo')
plt.colorbar()
plt.title("Pressure")



plt.show()