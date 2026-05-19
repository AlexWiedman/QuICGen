from QuICGen.r1calcFS import _residual, _jacobian
from QICalculator_in_General_Frame.QuICGen.quicgen import QIC
import numpy as np

rc = [1, -0.3]
zs = [0, -0.25]

test = QIC(rc, zs)

eps = 10**(-8)

x = np.full(test.nphi, 0.74)
x[0] = 0.

t_matrix = np.empty((test.nphi, test.nphi))

for idx in range(test.nphi):
    x0 = np.copy(x)
    x0[idx] = x0[idx] - eps
    t0 = _residual(test, x0)

    x1 = np.copy(x)
    x1[idx] = x1[idx] + eps
    t1 = _residual(test, x1)

    t_matrix[:, idx] = (t1 - t0) / (2 * eps)


jac = _jacobian(test, x)

print(np.max(np.abs(t_matrix - jac)))
