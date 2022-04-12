import numpy as np
import sys

cl = np.linspace(1.0, 26.0, 16)

for c in cl:
    sys.stdout.write('%0.1lf\t' % c)
print("")