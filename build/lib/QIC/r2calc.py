import numpy as np
from .util import mu0
"""
Calculate the Near Axis Expansion to second order 
"""
def r2calc(self):

    #shorthand terms
    nphi = self.nphi
    B0_over_abs_G0 = self.B0 / np.abs(self.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    X1c = self.X1c
    X1s = self.X1s
    Y1s = self.Y1s
    Y1c = self.Y1c
    sigma = self.sigma
    d_d_varphi = self.d_d_varphi
    iota_N = self.iotaN
    iota = self.iota
    k1 = self.k1
    k2 = self.k2
    k3 = self.k3
    B1s = self.B1s
    B1c = self.B1c
    B1c_prime = self.d_B1c_d_varphi
    B1s_prime = self.d_B1s_d_varphi
    B0 = self.B0
    G0 = self.G0
    I2 = self.I2
    B2s = self.B2s
    B2c = self.B2c
    p2 = self.p2
    sG = self.sG
    spsi = self.spsi
    I2_over_B0 = self.I2 / self.B0
    Bbar = self.Bbar
    dldp = self.d_l_d_varphi

    d_beta0_d_varphi = 2*mu0*p2*G0/Bbar * (1 / (B0 * B0) - 1 / (2*np.pi) * np.sum(1 / (B0 * B0)))
    beta0 = 2*mu0*p2*G0/Bbar * 1 / (2*np.pi) * np.sum(1 / (B0 * B0) - 1 / (2*np.pi) * np.sum(1 / (B0 * B0)))

    V1 = X1c * X1c + X1s * X1s + Y1c * Y1c + Y1s * Y1s
    V2 = 2 * (X1s * X1c + Y1s * Y1c)
    V3 = X1c * X1c - X1s * X1s + Y1c * Y1c - Y1s * Y1s

    Z20 = beta0 * Bbar / (2 * G0) * dldp - np.matmul(d_d_varphi,V1) / (8*dldp)
    Z2s = -1/(dldp*8)*(np.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    Z2c = -1/(dldp*8)*(np.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

    beta_1c = -4 * sG * mu0 * p2 * abs_G0_over_B0 / (iota_N * iota_N * Bbar * B0 * B0) * (B1c_prime - iota_N * B1s)
    beta_1s = -4 * sG * mu0 * p2 * abs_G0_over_B0 / (iota_N * iota_N * B0 * B0 * Bbar) * (B1s_prime + iota_N * B1c)

    qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * k3 * dldp
    qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * k3 * dldp
    rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * k3 * dldp
    rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * k3 * dldp

    eqA32_inhomogeneous = (sG * Bbar / (2 * B0)) * (X1s*k1 + Y1s*k2)
    eqA33_inhomogeneous = (sG * Bbar / (2 * B0)) * (X1c*k1 + Y1c*k2)

    eqA35_inhomogeneous = (1 / dldp) * (np.matmul(d_d_varphi, Z2s) - 2 * iota_N * Z2c \
                                        - (1 / dldp) * (-abs_G0_over_B0*abs_G0_over_B0*B2s/B0 + (3/2)*(abs_G0_over_B0*abs_G0_over_B0*B1c*B1s/(B0*B0)) \
                                        - (X1c*X1s*k1*k1 + X1c*Y1s*k1*k2 + X1s*Y1c*k1*k2 + Y1c*Y1s*k2*k2)*dldp*dldp/4  - (qc*qs+rc*rs)/2))
    eqA36_inhomogeneous = (1 / dldp) * (np.matmul(d_d_varphi, Z2c) + 2 * iota_N * Z2s \
                                        - (1 / dldp) * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0 + (3/4)*(abs_G0_over_B0*abs_G0_over_B0*(B1c*B1c-B1s*B1s)/(B0*B0)) \
                                        - ((X1c*X1c - X1s*X1s)*k1*k1 + 2*(X1c*Y1c-X1s*Y1s)*k1*k2 + (Y1c*Y1c - Y1s*Y1s)*k2*k2)*dldp*dldp/4  - (qc*qc - qs*qs + rc*rc - rs*rs)/4))

    fx0_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1c*Y1c+X1s*Y1s) + k2*(Y1c*Y1c+Y1s*Y1s)) * dldp - 0.5*beta0*k1*dldp*(X1s*Y1c - X1c*Y1s) - 0.5*dldp*(beta_1c*Y1s - beta_1s*Y1c) + k1*dldp*Z20
    fxs_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1s*Y1c+X1c*Y1s) + 2*k2*(Y1c*Y1s)) * dldp - 0.5*beta0*dldp*(k1*(X1c*Y1c - X1s*Y1s)-k2*(Y1s*Y1s-Y1c*Y1c)) - 0.5*dldp*(beta_1s*Y1s - beta_1c*Y1c) + k1*dldp*Z2s
    fxc_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1c*Y1c-X1s*Y1s) + k2*(Y1c*Y1c-Y1s*Y1s)) * dldp - 0.5*beta0*dldp*(-k1*(X1c*Y1s + X1s*Y1c)-2*k2*(Y1s*Y1c)) - 0.5*dldp*(beta_1c*Y1s + beta_1s*Y1c) + k1*dldp*Z2c

    fy0_inhomogeneous =  (I2/Bbar) * 0.5 * (k1*(X1s*X1s+X1c*X1c) + k2*(X1c*Y1c + X1s*Y1s)) * dldp - 0.5*beta0*dldp*(k2*(X1s*Y1c-X1c*Y1s)) - 0.5*dldp*(beta_1s*X1c - beta_1c*X1s) + k2*dldp*Z20
    fys_inhomogeneous =  (I2/Bbar) * 0.5 * (2*k1*(X1s*X1c) + k2*(X1s*Y1c + X1c*Y1s)) * dldp - 0.5*beta0*dldp*(k1*(X1s*X1s-X1c*X1c) - k2*(X1c*Y1c-X1s*Y1s)) - 0.5*dldp*(beta_1c*X1c - beta_1s*X1s) + k2*dldp*Z2s
    fyc_inhomogeneous =  (I2/Bbar) * 0.5 * (k1*(X1c*X1c - X1s*X1s) + k2*(X1c*Y1c - X1s*Y1s)) * dldp - 0.5*beta0*dldp*(2*k1*(X1s*X1c) + k2*(X1c*Y1s+X1s*Y1c)) + 0.5*dldp*(beta_1s*X1c + beta_1c*X1s) + k2*dldp*Z2c
    
    eqA41_inhomogeneous = -X1s*fx0_inhomogeneous + X1c*fxs_inhomogeneous - X1s*fxc_inhomogeneous - Y1s*fy0_inhomogeneous + Y1c*fys_inhomogeneous - Y1s*fyc_inhomogeneous
    eqA42_inhomogeneous = -X1c*fx0_inhomogeneous + X1s*fxs_inhomogeneous + X1c*fxc_inhomogeneous - Y1c*fy0_inhomogeneous + Y1s*fys_inhomogeneous + Y1c*fyc_inhomogeneous

    eqA32_X20 = -Y1c
    eqA32_X2s = Y1s
    eqA32_X2c = Y1c
    eqA32_Y20 = X1c
    eqA32_Y2s = -X1s
    eqA32_Y2c = -X1c

    eqA33_X20 = Y1s
    eqA33_X2s = -Y1c
    eqA33_X2c = Y1s
    eqA33_Y20 = -X1s
    eqA33_Y2s = X1c
    eqA33_Y2c = -X1s

    eqA35_X20 = np.zeros(nphi)
    eqA35_X2s = -k1
    eqA35_X2c = np.zeros(nphi)
    eqA35_Y20 = np.zeros(nphi)
    eqA35_Y2s = -k2
    eqA35_Y2c = np.zeros(nphi)

    eqA36_X20 = np.zeros(nphi)
    eqA36_X2s = np.zeros(nphi)
    eqA36_X2c = -k1
    eqA36_Y20 = np.zeros(nphi)
    eqA36_Y2s = np.zeros(nphi)
    eqA36_Y2c = -k2

    fx0_X20 = np.zeros(nphi)
    fx0_X2s = np.zeros(nphi)
    fx0_X2c = np.zeros(nphi)
    fx0_Y20 = -k3*dldp + 2*I2*dldp/Bbar
    fx0_Y2s =  4*G0*Z2c/Bbar
    fx0_Y2c = -4*G0*Z2s/Bbar

    fxs_X20 = np.zeros(nphi)
    fxs_X2s = np.zeros(nphi)
    fxs_X2c = -2*iota_N
    fxs_Y20 = 4*G0*Z2c/Bbar
    fxs_Y2s = -k3*dldp + 2*I2*dldp/Bbar
    fxs_Y2c = -4*G0*Z20/Bbar

    fxc_X20 = np.zeros(nphi)
    fxc_X2s = 2*iota_N
    fxc_X2c = np.zeros(nphi)
    fxc_Y20 = -4*G0*Z2s/Bbar
    fxc_Y2s = 4*G0*Z20/Bbar
    fxc_Y2c = -k3*dldp + 2*I2*dldp/Bbar
    
    fy0_X20 = k3*dldp - 2*I2*dldp/Bbar
    fy0_X2s = -4*G0*Z2c/Bbar
    fy0_X2c = 4*G0*Z2s/Bbar
    fy0_Y20 = np.zeros(nphi)
    fy0_Y2s = np.zeros(nphi)
    fy0_Y2c = np.zeros(nphi)
    
    fys_X20 = -4*G0*Z2c/Bbar
    fys_X2s = k3*dldp - 2*I2*dldp/Bbar
    fys_X2c = 4*G0*Z20/Bbar
    fys_Y20 = np.zeros(nphi)
    fys_Y2s = np.zeros(nphi)
    fys_Y2c = -2*iota_N

    fyc_X20 = 4*G0*Z2s/Bbar
    fyc_X2s = -4*G0*Z20/Bbar
    fyc_X2c = k3*dldp - 2*I2*dldp/Bbar
    fyc_Y20 = np.zeros(nphi)
    fyc_Y2s = 2*iota_N
    fyc_Y2c = np.zeros(nphi)

    eqA41_X20 = -X1s*fx0_X20 + X1c*fxs_X20 - X1s*fxc_X20 - Y1s*fy0_X20 + Y1c*fys_X20 - Y1s*fyc_X20
    eqA41_X2s = -X1s*fx0_X2s + X1c*fxs_X2s - X1s*fxc_X2s - Y1s*fy0_X2s + Y1c*fys_X2s - Y1s*fyc_X2s
    eqA41_X2c = -X1s*fx0_X2c + X1c*fxs_X2c - X1s*fxc_X2c - Y1s*fy0_X2c + Y1c*fys_X2c - Y1s*fyc_X2c
    eqA41_Y20 = -X1s*fx0_Y20 + X1c*fxs_Y20 - X1s*fxc_Y20 - Y1s*fy0_Y20 + Y1c*fys_Y20 - Y1s*fyc_Y20
    eqA41_Y2s = -X1s*fx0_Y2s + X1c*fxs_Y2s - X1s*fxc_Y2s - Y1s*fy0_Y2s + Y1c*fys_Y2s - Y1s*fyc_Y2s
    eqA41_Y2c = -X1s*fx0_Y2c + X1c*fxs_Y2c - X1s*fxc_Y2c - Y1s*fy0_Y2c + Y1c*fys_Y2c - Y1s*fyc_Y2c

    eqA42_X20 = -X1c*fx0_X20 + X1s*fxs_X20 + X1c*fxc_X20 - Y1c*fy0_X20 + Y1s*fys_X20 + Y1c*fyc_X20
    eqA42_X2s = -X1c*fx0_X2s + X1s*fxs_X2s + X1c*fxc_X2s - Y1c*fy0_X2s + Y1s*fys_X2s + Y1c*fyc_X2s
    eqA42_X2c = -X1c*fx0_X2c + X1s*fxs_X2c + X1c*fxc_X2c - Y1c*fy0_X2c + Y1s*fys_X2c + Y1c*fyc_X2c
    eqA42_Y20 = -X1c*fx0_Y20 + X1s*fxs_Y20 + X1c*fxc_Y20 - Y1c*fy0_Y20 + Y1s*fys_Y20 + Y1c*fyc_Y20
    eqA42_Y2s = -X1c*fx0_Y2s + X1s*fxs_Y2s + X1c*fxc_Y2s - Y1c*fy0_Y2s + Y1s*fys_Y2s + Y1c*fyc_Y2s
    eqA42_Y2c = -X1c*fx0_Y2c + X1s*fxs_Y2c + X1c*fxc_Y2c - Y1c*fy0_Y2c + Y1s*fys_Y2c + Y1c*fyc_Y2c

    matrix = np.zeros((6*nphi, 6*nphi))
    right_hand_side = np.zeros((6*nphi))
    for j in range(nphi):

        # Derivative terms
        #X20 terms
        matrix[j+4*nphi, 0*nphi:1*nphi] = -X1s[j] * d_d_varphi[j, :] #A41
        matrix[j+5*nphi, 0*nphi:1*nphi] = -X1c[j] * d_d_varphi[j, :] #A42
        #X2c terms
        matrix[j+4*nphi, 1*nphi:2*nphi] = -X1s[j] * d_d_varphi[j, :]
        matrix[j+5*nphi, 1*nphi:2*nphi] =  X1c[j] * d_d_varphi[j, :]
        #X2s terms
        matrix[j+4*nphi, 2*nphi:3*nphi] =  X1c[j] * d_d_varphi[j, :]
        matrix[j+5*nphi, 2*nphi:3*nphi] =  X1s[j] * d_d_varphi[j, :]
        #Y20 terms
        matrix[j+4*nphi, 3*nphi:4*nphi] = -Y1s[j] * d_d_varphi[j, :]
        matrix[j+5*nphi, 3*nphi:4*nphi] = -Y1c[j] * d_d_varphi[j, :]
        #Y2c terms
        matrix[j+4*nphi, 4*nphi:5*nphi] = -Y1s[j] * d_d_varphi[j, :]
        matrix[j+5*nphi, 4*nphi:5*nphi] =  Y1c[j] * d_d_varphi[j, :]
        #Y2s terms
        matrix[j+4*nphi, 5*nphi:6*nphi] =  Y1c[j] * d_d_varphi[j, :]
        matrix[j+5*nphi, 5*nphi:6*nphi] =  Y1s[j] * d_d_varphi[j, :]

        #Non derivatives

        #X20 terms
        matrix[j+0*nphi, j+0*nphi] = matrix[j+0*nphi, j+0*nphi] + eqA32_X20[j] #A32
        matrix[j+1*nphi, j+0*nphi] = matrix[j+1*nphi, j+0*nphi] + eqA33_X20[j] #A33
        matrix[j+2*nphi, j+0*nphi] = matrix[j+2*nphi, j+0*nphi] + eqA35_X20[j] #A35
        matrix[j+3*nphi, j+0*nphi] = matrix[j+3*nphi, j+0*nphi] + eqA36_X20[j] #A36
        matrix[j+4*nphi, j+0*nphi] = matrix[j+4*nphi, j+0*nphi] + eqA41_X20[j] #A41
        matrix[j+5*nphi, j+0*nphi] = matrix[j+5*nphi, j+0*nphi] + eqA42_X20[j] #A42
        #X2c terms
        matrix[j+0*nphi, j+1*nphi] = matrix[j+0*nphi, j+1*nphi] + eqA32_X2c[j]
        matrix[j+1*nphi, j+1*nphi] = matrix[j+1*nphi, j+1*nphi] + eqA33_X2c[j]
        matrix[j+2*nphi, j+1*nphi] = matrix[j+2*nphi, j+1*nphi] + eqA35_X2c[j]
        matrix[j+3*nphi, j+1*nphi] = matrix[j+3*nphi, j+1*nphi] + eqA36_X2c[j]
        matrix[j+4*nphi, j+1*nphi] = matrix[j+4*nphi, j+1*nphi] + eqA41_X2c[j]
        matrix[j+5*nphi, j+1*nphi] = matrix[j+5*nphi, j+1*nphi] + eqA42_X2c[j]
        #X2s terms
        matrix[j+0*nphi, j+2*nphi] = matrix[j+0*nphi, j+2*nphi] + eqA32_X2s[j]
        matrix[j+1*nphi, j+2*nphi] = matrix[j+1*nphi, j+2*nphi] + eqA33_X2s[j]
        matrix[j+2*nphi, j+2*nphi] = matrix[j+2*nphi, j+2*nphi] + eqA35_X2s[j]
        matrix[j+3*nphi, j+2*nphi] = matrix[j+3*nphi, j+2*nphi] + eqA36_X2s[j]
        matrix[j+4*nphi, j+2*nphi] = matrix[j+4*nphi, j+2*nphi] + eqA41_X2s[j]
        matrix[j+5*nphi, j+2*nphi] = matrix[j+5*nphi, j+2*nphi] + eqA42_X2s[j]
        #Y20 terms
        matrix[j+0*nphi, j+3*nphi] = matrix[j+0*nphi, j+3*nphi] + eqA32_Y20[j]
        matrix[j+1*nphi, j+3*nphi] = matrix[j+1*nphi, j+3*nphi] + eqA33_Y20[j]
        matrix[j+2*nphi, j+3*nphi] = matrix[j+2*nphi, j+3*nphi] + eqA35_Y20[j]
        matrix[j+3*nphi, j+3*nphi] = matrix[j+3*nphi, j+3*nphi] + eqA36_Y20[j]
        matrix[j+4*nphi, j+3*nphi] = matrix[j+4*nphi, j+3*nphi] + eqA41_Y20[j]
        matrix[j+5*nphi, j+3*nphi] = matrix[j+5*nphi, j+3*nphi] + eqA42_Y20[j]
        #Y2c terms
        matrix[j+0*nphi, j+4*nphi] = matrix[j+0*nphi, j+4*nphi] + eqA32_Y2c[j]
        matrix[j+1*nphi, j+4*nphi] = matrix[j+1*nphi, j+4*nphi] + eqA33_Y2c[j]
        matrix[j+2*nphi, j+4*nphi] = matrix[j+2*nphi, j+4*nphi] + eqA35_Y2c[j]
        matrix[j+3*nphi, j+4*nphi] = matrix[j+3*nphi, j+4*nphi] + eqA36_Y2c[j]
        matrix[j+4*nphi, j+4*nphi] = matrix[j+4*nphi, j+4*nphi] + eqA41_Y2c[j]
        matrix[j+5*nphi, j+4*nphi] = matrix[j+5*nphi, j+4*nphi] + eqA42_Y2c[j]
        #Y2s terms
        matrix[j+0*nphi, j+5*nphi] = matrix[j+0*nphi, j+5*nphi] + eqA32_Y2s[j]
        matrix[j+1*nphi, j+5*nphi] = matrix[j+1*nphi, j+5*nphi] + eqA33_Y2s[j]
        matrix[j+2*nphi, j+5*nphi] = matrix[j+2*nphi, j+5*nphi] + eqA35_Y2s[j]
        matrix[j+3*nphi, j+5*nphi] = matrix[j+3*nphi, j+5*nphi] + eqA36_Y2s[j]
        matrix[j+4*nphi, j+5*nphi] = matrix[j+4*nphi, j+5*nphi] + eqA41_Y2s[j]
        matrix[j+5*nphi, j+5*nphi] = matrix[j+5*nphi, j+5*nphi] + eqA42_Y2s[j]


    #Inhomogeneous
    right_hand_side[0*nphi:1*nphi] = -(eqA32_inhomogeneous)
    right_hand_side[1*nphi:2*nphi] = -(eqA33_inhomogeneous)
    right_hand_side[2*nphi:3*nphi] = -(eqA35_inhomogeneous)
    right_hand_side[3*nphi:4*nphi] = -(eqA36_inhomogeneous)
    right_hand_side[4*nphi:5*nphi] = -(eqA41_inhomogeneous)
    right_hand_side[5*nphi:6*nphi] = -(eqA42_inhomogeneous)

    
    solution = np.linalg.solve(matrix, right_hand_side)


    X20 = solution[0*nphi:1*nphi]
    X2c = solution[1*nphi:2*nphi]
    X2s = solution[2*nphi:3*nphi]
    Y20 = solution[3*nphi:4*nphi]
    Y2c = solution[4*nphi:5*nphi]
    Y2s = solution[5*nphi:6*nphi]

    """
    A41derivterms = -X1s * np.matmul(d_d_varphi,X20) - X1s * np.matmul(d_d_varphi, X2c) + X1c * np.matmul(d_d_varphi, X2s) - Y1s * np.matmul(d_d_varphi, Y20) - Y1s * np.matmul(d_d_varphi, Y2c) + Y1c * np.matmul(d_d_varphi, Y2s)
    A42derivterms = -X1c * np.matmul(d_d_varphi,X20) + X1c * np.matmul(d_d_varphi, X2c) + X1s * np.matmul(d_d_varphi, X2s) - Y1c * np.matmul(d_d_varphi, Y20) + Y1c * np.matmul(d_d_varphi, Y2c) + Y1s * np.matmul(d_d_varphi, Y2s)

    eqA32is0 = eqA32_inhomogeneous + eqA32_X20 * X20 + eqA32_X2c * X2c + eqA32_X2s * X2s + eqA32_Y20 * Y20 + eqA32_Y2c * Y2c + eqA32_Y2s * Y2s
    eqA33is0 = eqA33_inhomogeneous + eqA33_X20 * X20 + eqA33_X2c * X2c + eqA33_X2s * X2s + eqA33_Y20 * Y20 + eqA33_Y2c * Y2c + eqA33_Y2s * Y2s
    eqA35is0 = eqA35_inhomogeneous + eqA35_X20 * X20 + eqA35_X2c * X2c + eqA35_X2s * X2s + eqA35_Y20 * Y20 + eqA35_Y2c * Y2c + eqA35_Y2s * Y2s
    eqA36is0 = eqA36_inhomogeneous + eqA36_X20 * X20 + eqA36_X2c * X2c + eqA36_X2s * X2s + eqA36_Y20 * Y20 + eqA36_Y2c * Y2c + eqA36_Y2s * Y2s
    eqA41is0 = eqA41_inhomogeneous + eqA41_X20 * X20 + eqA41_X2c * X2c + eqA41_X2s * X2s + eqA41_Y20 * Y20 + eqA41_Y2c * Y2c + eqA41_Y2s * Y2s + A41derivterms
    eqA42is0 = eqA42_inhomogeneous + eqA42_X20 * X20 + eqA42_X2c * X2c + eqA42_X2s * X2s + eqA42_Y20 * Y20 + eqA42_Y2c * Y2c + eqA42_Y2s * Y2s + A42derivterms

    np.testing.assert_allclose(np.sum(eqA32is0), 0, atol=1E-7)
    np.testing.assert_allclose(np.sum(eqA33is0), 0, atol=1E-7)
    np.testing.assert_allclose(np.sum(eqA35is0), 0, atol=1E-7)
    np.testing.assert_allclose(np.sum(eqA36is0), 0, atol=1E-7)
    np.testing.assert_allclose(np.sum(eqA41is0), 0, atol=1E-7)
    np.testing.assert_allclose(np.sum(eqA42is0), 0, atol=1E-7)
    """
    
    B20 = (B0*B0_over_abs_G0*B0_over_abs_G0) * (dldp*((X20 * k1 + Y20 * k2) * dldp - np.matmul(d_d_varphi, Z20)) \
          + (0.75*abs_G0_over_B0*abs_G0_over_B0*(B1c*B1c+B1s*B1s)/(B0*B0)) + (abs_G0_over_B0*abs_G0_over_B0*(-mu0 * p2))/(B0*B0) \
          - 0.25 * dldp * dldp * ((X1c * k1 + Y1c * k2)*(X1c * k1 + Y1c * k2) + (X1s * k1 + Y1s * k2)*(X1s * k1 + Y1s * k2)) \
          - 0.25 * (qc * qc + qs * qs + rc * rc + rs * rs))    
          

    d_l_d_phi = self.d_l_d_phi
    normalizer = 1 / np.sum(d_l_d_phi)
    self.B20_mean = np.sum(B20 * d_l_d_phi) * normalizer
    self.B20_anomaly = B20 - self.B20_mean
    self.B20_residual = np.sqrt(np.sum((B20 - self.B20_mean) * (B20 - self.B20_mean) * d_l_d_phi) * normalizer) / B0
    self.B20_variation = np.max(B20) - np.min(B20)

    self.N_helicity = - self.helicity * self.nfp
    self.G2 = -mu0 * p2 * G0 / (B0 * B0) - iota * I2

    self.d_k1_d_varphi = np.matmul(d_d_varphi, k1)
    self.d_k2_d_varphi = np.matmul(d_d_varphi, k2)
    self.d_k3_d_varphi = np.matmul(d_d_varphi, k3)
    self.d_X20_d_varphi = np.matmul(d_d_varphi, X20)
    self.d_X2s_d_varphi = np.matmul(d_d_varphi, X2s)
    self.d_X2c_d_varphi = np.matmul(d_d_varphi, X2c)
    self.d_Y20_d_varphi = np.matmul(d_d_varphi, Y20)
    self.d_Y2s_d_varphi = np.matmul(d_d_varphi, Y2s)
    self.d_Y2c_d_varphi = np.matmul(d_d_varphi, Y2c)
    self.d_Z20_d_varphi = np.matmul(d_d_varphi, Z20)
    self.d_Z2s_d_varphi = np.matmul(d_d_varphi, Z2s)
    self.d_Z2c_d_varphi = np.matmul(d_d_varphi, Z2c)
    self.d2_X1c_d_varphi2 = np.matmul(d_d_varphi, self.d_X1c_d_varphi)
    self.d2_Y1c_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1c_d_varphi)
    self.d2_Y1s_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1s_d_varphi)

    # Store all important results in self:
    self.V1 = V1
    self.V2 = V2
    self.V3 = V3

    self.X20 = X20
    self.X2s = X2s
    self.X2c = X2c
    self.Y20 = Y20
    self.Y2s = Y2s
    self.Y2c = Y2c
    self.Z20 = Z20
    self.Z2s = Z2s
    self.Z2c = Z2c
    self.beta0 = beta0
    self.d_beta0_d_varphi = d_beta0_d_varphi
    self.beta_1s = beta_1s
    self.beta_1c = beta_1c
    self.B20 = B20

    # Since centroid doesn't twist, we ignore for now. This should be updated when helicity needs to be taken into account
    if self.helicity == 0:
        self.X20_untwisted = self.X20
        self.X2s_untwisted = self.X2s
        self.X2c_untwisted = self.X2c

        self.Y20_untwisted = self.Y20
        self.Y2s_untwisted = self.Y2s
        self.Y2c_untwisted = self.Y2c

        self.Z20_untwisted = self.Z20
        self.Z2s_untwisted = self.Z2s
        self.Z2c_untwisted = self.Z2c

        self.B20_untwisted = self.B20
        self.B2s_untwisted = self.B2s
        self.B2c_untwisted = self.B2c
    else:
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        self.X20_untwisted = self.X20
        self.Y20_untwisted = self.Y20
        self.Z20_untwisted = self.Z20
        self.B20_untwisted = self.B20
        sinangle = np.sin(2*angle)
        cosangle = np.cos(2*angle)
        self.X2s_untwisted = self.X2s *   cosangle  + self.X2c * sinangle
        self.X2c_untwisted = self.X2s * (-sinangle) + self.X2c * cosangle
        self.Y2s_untwisted = self.Y2s *   cosangle  + self.Y2c * sinangle
        self.Y2c_untwisted = self.Y2s * (-sinangle) + self.Y2c * cosangle
        self.Z2s_untwisted = self.Z2s *   cosangle  + self.Z2c * sinangle
        self.Z2c_untwisted = self.Z2s * (-sinangle) + self.Z2c * cosangle
        self.B2s_untwisted = self.B2s *   cosangle  + self.B2c * sinangle
        self.B2c_untwisted = self.B2s * (-sinangle) + self.B2c * cosangle

    # O(r^2) diagnostics:
    #self.mercier()
    #self.calculate_grad_grad_B_tensor()
    #self.grad_grad_B_inverse_scale_length_vs_varphi = t.grad_grad_B_inverse_scale_length_vs_varphi
    #self.grad_grad_B_inverse_scale_length = t.grad_grad_B_inverse_scale_length
    self.calculate_r_singularity()