import numpy as np
import time
from QIC.newtonMethod import newton
"""
Solve the Near Axis Expansion to first order using the Lambda equation.
"""

def _residual(self, x):
    """
    Calculate the residual of the lambda equation (called sigma here, since lambda is a python command)
    """
    sigma = np.copy(x)
    sigma[0] = self.sigma0
    iotaN = x[0] #+ self.torsion_sign_change * self.nfp
    # iotaN = iota + self.helicity * self.nfp

    L2 = np.power(self.X1c,2) + np.power(self.Y1c,2)
    beta = -1 * self.Bbar / (self.B0 * L2)

    X1cP = np.matmul(self.d_d_varphi, self.X1c)
    Y1cP = np.matmul(self.d_d_varphi, self.Y1c)
    B0P = np.matmul(self.d_d_varphi, self.B0)

    r = np.matmul(self.d_d_varphi, sigma) \
        - sigma * ((2 * (self.X1c * X1cP + self.Y1c * Y1cP)) / L2 + B0P / self.B0) \
        + beta * iotaN * (sigma * sigma + 1 + 1 / (beta * beta))  \
        + 2 * self.sG * (self.Y1c * X1cP - self.X1c * Y1cP) / L2 \
        + 2 * self.G0 / self.B0 * (self.I2 / self.Bbar - self.k3)
    return r

def _jacobian(self, x):
    """
    Calculate the jacobian of the lambda equation
    """
    sigma = np.copy(x)
    sigma[0] = self.sigma0
    iotaN = x[0] #+ self.torsion_sign_change * self.nfp
    # iotaN = iota + self.helicity * self.nfp

    X1cP = np.matmul(self.d_d_varphi, self.X1c)
    Y1cP = np.matmul(self.d_d_varphi, self.Y1c)
    B0P = np.matmul(self.d_d_varphi, self.B0)

    L2 = np.power(self.X1c,2) + np.power(self.Y1c,2)
    beta = -1 * self.Bbar / (self.B0 * L2)

    jac = np.copy(self.d_d_varphi)
    #dr/d(sigma)
    for j in range(self.nphi):
        jac[j, j] += iotaN * 2 * beta[j] * sigma[j] - ((2 * (self.X1c[j] * X1cP[j] + self.Y1c[j] * Y1cP[j])) / L2[j] + B0P[j] / self.B0[j])
    #dr/d(iota)
    jac[:, 0] = beta * (sigma * sigma + 1 + 1 / (beta * beta))

    return jac

def solve_sigma_equation(self):
    
    x0 = np.full(self.nphi, self.sigma0)
    x0[0] = 0 # Initial guess for iota

    self.sigma = newton(self._residual, x0, jac=self._jacobian)
    self.iotaN = self.sigma[0] #+ self.torsion_sign_change * self.nfp
    # self.iotaN = self.sigma[0] + self.helicity * self.nfp + self.torsion_sign_change * self.nfp
    self.sigma[0] = self.sigma0

def _determine_helicity_centroid(self):
    """
    This helicity calculation is based on changes in sign of X1c and Y1c
    This will only work for the centroid frame, or other frames where a vector will point outwards radially
    It is not analagous to the helicity of the FS frame
    """

    quadrant = np.zeros(self.nphi + 1)
    for j in range(self.nphi):
        if self.X1c[j] >= 0:
            if self.X1s[j] >= 0:
                quadrant[j] = 1
            else:
                quadrant[j] = 4
        else:
            if self.X1s[j] >= 0:
                quadrant[j] = 2
            else:
                quadrant[j] = 3

    quadrant[self.nphi] = quadrant[0]

    counter = 0
    for j in range(self.nphi):
        if quadrant[j] == 4 and quadrant[j+1] == 1:
            counter += 1
        elif quadrant[j] == 1 and quadrant[j+1] == 4:
            counter -= 1
        else:
            counter += quadrant[j+1] - quadrant[j]

    counter *= self.spsi * self.sG
    self.helicity = -counter / 4

def r1_diagnostics(self):
    """
    Find other first order properties, after sigma and iota are found.
    """

    L2 = np.power(self.X1c,2) + np.power(self.Y1c,2)
    beta = -1 * self.Bbar / (self.B0 * L2)

    self.Y1s = -beta * (self.X1c + self.Y1c * self.sigma)
    self.X1s = -beta * (np.multiply(-1, self.Y1c) + self.X1c * self.sigma)

    self._determine_helicity_centroid()
    self.iota = self.iotaN + self.helicity * self.nfp



    self.B1c = (self.X1c * self.k1 + self.Y1c * self.k2) * self.B0 #* self.sign_curvature_change
    self.B1s = (self.X1s * self.k1 + self.Y1s * self.k2) * self.B0 #* self.sign_curvature_change

    if self.helicity == 0:
        self.X1s_untwisted = self.X1s
        self.X1c_untwisted = self.X1c
        self.Y1s_untwisted = self.Y1s
        self.Y1c_untwisted = self.Y1c
        self.B1s_untwisted = self.B1s
        self.B1c_untwisted = self.B1c
    else:
        angle = self.helicity * self.varphi * self.nfp
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        self.X1s_untwisted = self.X1s *   cosangle  + self.X1c * sinangle
        self.X1c_untwisted = self.X1s * (-sinangle) + self.X1c * cosangle
        self.Y1s_untwisted = self.Y1s *   cosangle  + self.Y1c * sinangle
        self.Y1c_untwisted = self.Y1s * (-sinangle) + self.Y1c * cosangle
        self.B1s_untwisted = self.B1s *   cosangle  + self.B1c * sinangle
        self.B1c_untwisted = self.B1s * (-sinangle) + self.B1c * cosangle
    
    self.d_X1c_d_varphi = np.matmul(self.d_d_varphi, self.X1c)
    self.d_X1s_d_varphi = np.matmul(self.d_d_varphi, self.X1s)
    self.d_Y1s_d_varphi = np.matmul(self.d_d_varphi, self.Y1s)
    self.d_Y1c_d_varphi = np.matmul(self.d_d_varphi, self.Y1c)
    self.d_B1c_d_varphi = np.matmul(self.d_d_varphi, self.B1c)
    self.d_B1s_d_varphi = np.matmul(self.d_d_varphi, self.B1s)

    self.d_B0_d_varphi = np.matmul(self.d_d_varphi, self.B0)
    self.d_2_B0_d_varphi_2 = np.matmul(self.d_d_varphi, self.d_B0_d_varphi)