# Just a quick QIC test

from QICalculator_in_General_Frame.QuICGen.quicgen import QIC
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

nphi = 21

rc = np.zeros(5)
zc = np.zeros(5)
rs = np.zeros(5)
zs = np.zeros(5)
X1c = np.ones(nphi)
Y1c = np.ones(nphi)
B0 = np.ones(nphi)

rc[0] = 1
rc[2] = 0.34
zs[2] = 0.34

test = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nphi=nphi, B0=B0)

test.calculate()

print(test.k1)
print(test.k2)
print(test.k3)