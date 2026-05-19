import numpy as np
import time
from QuICGen.newtonMethod import newton
"""
Solve the Near Axis Expansion to first order with a new sigma equation. 
This solves a slightly different equation than r1calcLambda but the inputs and outputs are identical
"""
def _residual(self, x):
    """
    Calculate the residual of the sigma equation
    """
    sigma = np.copy(x)
    sigma[0] = self.sigma0
    iotaN = x[0]

    B1cP = np.matmul(self.d_d_varphi, self.B1c)
    B1sP = np.matmul(self.d_d_varphi, self.B1s)
    S = self.B1c*self.B1c + self.B1s*self.B1s
    D = self.B1c*self.B1c - self.B1s*self.B1s
    P = self.B1c*self.B1s
    W = B1cP*self.B1s - B1sP*self.B1c
    omega = W/S
    M = self.B0*self.Bbar*self.k1
    H = M*(self.k1*self.k1+self.k2*self.k2)
    a = H + self.k2*D
    q = H*(iotaN+sigma)
    r = self.k2/2 * (np.matmul(self.d_d_varphi, D) + np.matmul(self.d_d_varphi, S) / S) - 4 * self.k2 * P * iotaN
    t = iotaN * (H + 2*S*self.k2 + S*S/M) + omega*(H+S*S/M) - 2*(self.I2/self.Bbar - self.k3)*self.G0*self.k1*S/self.B0

    res = a*np.matmul(self.d_d_varphi, sigma) + q*sigma*sigma + r*sigma + t

    return res
    

def _jacobian(self, x):
    """
    Calculate the jacobian of the sigma equation
    """
    sigma = np.copy(x)
    sigma[0] = self.sigma0
    iotaN = x[0]

    B1cP = np.matmul(self.d_d_varphi, self.B1c)
    B1sP = np.matmul(self.d_d_varphi, self.B1s)
    S = self.B1c*self.B1c + self.B1s*self.B1s
    D = self.B1c*self.B1c - self.B1s*self.B1s
    P = self.B1c*self.B1s
    W = B1cP*self.B1s - B1sP*self.B1c
    omega = W/S
    M = self.B0*self.Bbar*self.k1
    H = M*(self.k1*self.k1+self.k2*self.k2)
    a = H + self.k2*D
    q = H*(iotaN+sigma)
    r = self.k2/2 * (np.matmul(self.d_d_varphi, D) + np.matmul(self.d_d_varphi, S) / S) - 4 * self.k2 * P * iotaN
    t = iotaN * (H + 2*S*self.k2 + S*S/M) + omega*(H+S*S/M) - 2*(self.I2/self.Bbar - self.k3)*self.G0*self.k1*S/self.B0

    jac = a * np.copy(self.d_d_varphi)
    

    #dr/d(sigma)
    for j in range(self.nphi):
        jac[j, j] += 2 * q * sigma[j] + r
    #dr/d(iota)
    jac [:, 0] = H*sigma*sigma - 4*self.k2*P*sigma + H + 2*self.k2*S+S*S/M

    return jac
    

def solve_sigma_equation(self):
    x0 = np.full(self.nphi, self.sigma0)
    x0[0] = 0 # Initial guess for iota

    self.sigma = newton(self._residual, x0, jac=self._jacobian)
    self.iota = self.sigma[0]
    self.iotaN = self.iota
    self.sigma[0] = self.sigma0
    
    

def r1_diagnostics(self):
    """
    Find other first order properties, after sigma and iota are found.
    """

    L2 = np.power(self.X1c,2) + np.power(self.Y1c,2)

    self.Y1c = (self.B1c+self.B1s*self.sigma)*self.Bbar*self.k1/ (self.B1s*self.B1s + self.B1c*self.B1c)
    self.Y1s = (-self.B1s+self.B1c*self.sigma)*self.Bbar*self.k1/ (self.B1s*self.B1s + self.B1c*self.B1c)

    self.X1c = (self.B1c - self.Y1c*self.k2) / self.k1
    self.X1s = (self.B1s - self.Y1s*self.k2) / self.k1

    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle that
    # is actually helical, with the theta=0 curve wrapping around the magnetic axis as you follow phi around toroidally. Therefore
    # here we convert to an untwisted poloidal angle, such that the theta=0 curve does not wrap around the axis.
    # helicity stuff should be the same as in pyQSC, not affected by the change to equations.
    if self.helicity == 0:
        self.X1s_untwisted = self.X1s
        self.X1c_untwisted = self.X1c
        self.Y1s_untwisted = self.Y1s
        self.Y1c_untwisted = self.Y1c
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        self.X1s_untwisted = self.X1s *   cosangle  + self.X1c * sinangle
        self.X1c_untwisted = self.X1s * (-sinangle) + self.X1c * cosangle
        self.Y1s_untwisted = self.Y1s *   cosangle  + self.Y1c * sinangle
        self.Y1c_untwisted = self.Y1s * (-sinangle) + self.Y1c * cosangle
    
    self.d_X1c_d_varphi = np.matmul(self.d_d_varphi, self.X1c)
    self.d_X1s_d_varphi = np.matmul(self.d_d_varphi, self.X1s)
    self.d_Y1s_d_varphi = np.matmul(self.d_d_varphi, self.Y1s)
    self.d_Y1c_d_varphi = np.matmul(self.d_d_varphi, self.Y1c)
