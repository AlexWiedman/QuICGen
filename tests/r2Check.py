import numpy as np
import logging
from QICalculator_in_General_Frame.QuICGen.quicgen import QIC
from QuICGen.centroidFrame import get_centroid_X1_Y1, get_centroid_X2_Y2
from QuICGen.util import cylindrical_to_centroid
from QuICGen.init_axis import init_axis
from qsc.qsc import Qsc

nfourier = 5
nphi = 61
sigma0 = 0
B2c = -0.00322
nfp = 2

rc = np.zeros(nfourier)
zs = np.zeros(nfourier)
rc[0] = 1
rc[1] = 0.155
rc[2] = 0.0102
zs[0] = 0
zs[1] = 0.154
zs[2] = 0.0111

qsc_stel = Qsc(rc, zs, nphi=nphi, order='r2', sigma0=sigma0, etabar=0.64, B2c = B2c, nfp=nfp)
B0 = np.full(qsc_stel.nphi, qsc_stel.B0)
X1c = qsc_stel.X1c
Y1c = qsc_stel.Y1c

qic_stel = QIC(rc, zs, X1c=X1c, Y1c=Y1c, B0=B0, order="r2", sigma0=sigma0, B2c=np.full(nphi, B2c), nfp=nfp, frame='FS')


d_d_varphi = qic_stel.d_d_varphi
dldp = qic_stel.d_l_d_varphi
Z2s = qic_stel.Z2s
Z2c = qic_stel.Z2c
Z20 = qsc_stel.Z20
B0_over_abs_G0 = qic_stel.B0 / np.abs(qic_stel.G0)
B2s = qic_stel.B2s
B2c = qic_stel.B2c
B1c = qic_stel.B1c
B1s = qic_stel.B1s
X1s = qic_stel.X1s
X1c = qic_stel.X1c
Y1s = qic_stel.Y1s
Y1c = qic_stel.Y1c
k1 = qic_stel.k1
k2 = qic_stel.k2
k3 = qic_stel.k3
abs_G0_over_B0 = 1 / B0_over_abs_G0
iota_N = qic_stel.iotaN
sG = qic_stel.sG
Bbar = qic_stel.Bbar
I2 = qsc_stel.I2
beta0 = 0
beta_1c = 0
beta_1s = qsc_stel.beta_1s
G0 = qic_stel.G0

X20 = qic_stel.X20
X2s = qic_stel.X2s
X2c = qic_stel.X2c
Y20 = qic_stel.Y20
Y2s = qic_stel.Y2s
Y2c = qic_stel.Y2c

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

fx0_qic = fx0_X20*X20 + fx0_X2s*X2s + fx0_X2c*X2c + fx0_Y20*Y20 + fx0_Y2s*Y2s #+ fx0_Y2c*Y2c# + np.matmul(d_d_varphi, X20)
fxs_qic = fxs_X20*X20 + fxs_X2s*X2s + fxs_X2c*X2c + fxs_Y20*Y20 + fxs_Y2s*Y2s #+ fxs_Y2c*Y2c# + np.matmul(d_d_varphi, X2s)
fxc_qic = fxc_X20*X20 + fxc_X2s*X2s + fxc_X2c*X2c + fxc_Y20*Y20 + fxc_Y2s*Y2s #+ fxc_Y2c*Y2c# + np.matmul(d_d_varphi, X2c)

fy0_qic = fy0_X20*X20 + fy0_X2s*X2s + fy0_X2c*X2c + fy0_Y20*Y20 + fy0_Y2s*Y2s #+ fy0_Y2c*Y2c# + np.matmul(d_d_varphi, Y20)
fys_qic = fys_X20*X20 + fys_X2s*X2s + fys_X2c*X2c + fys_Y20*Y20 + fys_Y2s*Y2s #+ fys_Y2c*Y2c# + np.matmul(d_d_varphi, Y2s)
fyc_qic = fyc_X20*X20 + fyc_X2s*X2s + fyc_X2c*X2c + fyc_Y20*Y20 + fyc_Y2s*Y2s #+ fyc_Y2c*Y2c# + np.matmul(d_d_varphi, Y2c)

eqA41_X20 = -X1s*fx0_X20 + X1c*fxs_X20 - X1s*fxc_X20 - Y1s*fy0_X20 + Y1c*fys_X20 - Y1s*fyc_X20
eqA41_X2s = -X1s*fx0_X2s + X1c*fxs_X2s - X1s*fxc_X2s - Y1s*fy0_X2s + Y1c*fys_X2s - Y1s*fyc_X2s
eqA41_X2c = -X1s*fx0_X2c + X1c*fxs_X2c - X1s*fxc_X2c - Y1s*fy0_X2c + Y1c*fys_X2c - Y1s*fyc_X2c
eqA41_Y20 = -X1s*fx0_Y20 + X1c*fxs_Y20 - X1s*fxc_Y20 - Y1s*fy0_Y20 + Y1c*fys_Y20 - Y1s*fyc_Y20
eqA41_Y2s = -X1s*fx0_Y2s + X1c*fxs_Y2s - X1s*fxc_Y2s - Y1s*fy0_Y2s + Y1c*fys_Y2s - Y1s*fyc_Y2s
eqA41_Y2c = -X1s*fx0_Y2c + X1c*fxs_Y2c - X1s*fxc_Y2c - Y1s*fy0_Y2c + Y1c*fys_Y2c - Y1s*fyc_Y2c

A41derivterms = -X1s * np.matmul(d_d_varphi,X20) - X1s * np.matmul(d_d_varphi, X2c) + X1c * np.matmul(d_d_varphi, X2s) - Y1s * np.matmul(d_d_varphi, Y20) - Y1s * np.matmul(d_d_varphi, Y2c) + Y1c * np.matmul(d_d_varphi, Y2s)
eqA41_qic = eqA41_X20 * X20 + eqA41_X2c * X2c + eqA41_X2s * X2s + eqA41_Y20 * Y20 + eqA41_Y2s * Y2s #+ eqA41_Y2c * Y2c # + A41derivterms

eqA41_qic_alt = -X1s*fx0_qic + X1c*fxs_qic - X1s*fxc_qic - Y1s*fy0_qic + Y1c*fys_qic - Y1s*fyc_qic


qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * k3 * dldp
qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * k3 * dldp
rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * k3 * dldp
rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * k3 * dldp

eqA32_inhomogeneous_qic = (sG * Bbar / (2 * B0)) * (X1s*k1 + Y1s*k2)
eqA33_inhomogeneous_qic = (sG * Bbar / (2 * B0)) * (X1c*k1 + Y1c*k2)
eqA32_qic = -Y1c*X20 + Y1s*X2s + Y1c*X2c + X1c*Y20 - X1s*Y2s - X1c*Y2c
eqA33_qic = Y1s*X20 - Y1c*X2s + Y1s*X2c - X1s*Y20 + X1c*Y2s - X1s*Y2c

eqA35_inhomogeneous_qic = (1 / dldp) * (np.matmul(d_d_varphi, Z2s) - 2 * iota_N * Z2c \
                                        - (1 / dldp) * (-abs_G0_over_B0*abs_G0_over_B0*B2s/B0 + (3/2)*(abs_G0_over_B0*abs_G0_over_B0*B1c*B1s/(B0*B0)) \
                                        - (X1c*X1s*k1*k1 + X1c*Y1s*k1*k2 + X1s*Y1c*k1*k2 + Y1c*Y1s*k2*k2)*dldp*dldp/4  - (qc*qs+rc*rs)/2))

eqA36_inhomogeneous_qic = (1 / dldp) * (np.matmul(d_d_varphi, Z2c) + 2 * iota_N * Z2s \
                                        - (1 / dldp) * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0 + (3/4)*(abs_G0_over_B0*abs_G0_over_B0*(B1c*B1c-B1s*B1s)/(B0*B0)) \
                                        - ((X1c*X1c - X1s*X1s)*k1*k1 + 2*(X1c*Y1c-X1s*Y1s)*k1*k2 + (Y1c*Y1c - Y1s*Y1s)*k2*k2)*dldp*dldp/4  - (qc*qc - qs*qs + rc*rc - rs*rs)/4))

fx0_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1c*Y1c+X1s*Y1s) + k2*(Y1c*Y1c+Y1s*Y1s)) * dldp - 0.5*beta0*k1*dldp*(X1s*Y1c - X1c*Y1s) - 0.5*dldp*(beta_1c*Y1s - beta_1s*Y1c) + k1*dldp*Z20
fxs_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1s*Y1c+X1c*Y1s) + 2*k2*(Y1c*Y1s)) * dldp - 0.5*beta0*dldp*(k1*(X1c*Y1c - X1s*Y1s)-k2*(Y1s*Y1s-Y1c*Y1c)) - 0.5*dldp*(beta_1s*Y1s - beta_1c*Y1c) + k1*dldp*Z2s
fxc_inhomogeneous = -(I2/Bbar) * 0.5 * (k1*(X1c*Y1c-X1s*Y1s) + k2*(Y1c*Y1c-Y1s*Y1s)) * dldp - 0.5*beta0*dldp*(-k1*(X1c*Y1s + X1s*Y1c)-2*k2*(Y1s*Y1c)) - 0.5*dldp*(beta_1c*Y1s + beta_1s*Y1c) + k1*dldp*Z2c

fy0_inhomogeneous =  (I2/Bbar) * 0.5 * (k1*(X1s*X1s+X1c*X1c) + k2*(X1c*Y1c + X1s*Y1s)) * dldp - 0.5*beta0*dldp*(k2*(X1s*Y1c-X1c*Y1s)) - 0.5*dldp*(beta_1s*X1c - beta_1c*X1s) + k2*dldp*Z20
fys_inhomogeneous =  (I2/Bbar) * 0.5 * (2*k1*(X1s*X1c) + k2*(X1s*Y1c + X1c*Y1s)) * dldp - 0.5*beta0*dldp*(k1*(X1s*X1s-X1c*X1c) - k2*(X1c*Y1c-X1s*Y1s)) - 0.5*dldp*(beta_1c*X1c - beta_1s*X1s) + k2*dldp*Z2s
fyc_inhomogeneous =  (I2/Bbar) * 0.5 * (k1*(X1c*X1c - X1s*X1s) + k2*(X1c*Y1c - X1s*Y1s)) * dldp - 0.5*beta0*dldp*(2*k1*(X1s*X1c) + k2*(X1c*Y1s+X1s*Y1c)) + 0.5*dldp*(beta_1s*X1c + beta_1c*X1s) + k2*dldp*Z2c

eqA41_inhomogeneous_qic = -X1s*fx0_inhomogeneous + X1c*fxs_inhomogeneous - X1s*fxc_inhomogeneous - Y1s*fy0_inhomogeneous + Y1c*fys_inhomogeneous - Y1s*fyc_inhomogeneous
eqA42_inhomogeneous_qic = -X1c*fx0_inhomogeneous + X1s*fxs_inhomogeneous + X1c*fxc_inhomogeneous - Y1c*fy0_inhomogeneous + Y1s*fys_inhomogeneous + Y1c*fyc_inhomogeneous


if False:
    X1c = qic_stel.X1c
    Y1c = qic_stel.Y1c
    Y1s = qic_stel.Y1s
    X1s = qic_stel.X1s

    X20 = qic_stel.X20
    X2s = qic_stel.X2s
    X2c = qic_stel.X2c
    Y20 = qic_stel.Y20
    Y2s = qic_stel.Y2s
    Y2c = qic_stel.Y2c
else:
    X1c = qsc_stel.X1c
    Y1c = qsc_stel.Y1c
    Y1s = qsc_stel.Y1s
    X1s = qsc_stel.X1s

    X20 = qsc_stel.X20
    X2s = qsc_stel.X2s
    X2c = qsc_stel.X2c
    Y20 = qsc_stel.Y20
    Y2s = qsc_stel.Y2s
    Y2c = qsc_stel.Y2c

fx0_qsc = np.matmul(d_d_varphi,X20) + (-k3 * dldp + 2*I2*dldp/Bbar) * Y20 + 4*G0*Z2c/Bbar * Y2s - 4*G0*Z2s/Bbar * Y2c
fxs_qsc = np.matmul(d_d_varphi,X2s) -2*iota_N*X2c + 4*G0*Z2c/Bbar*Y20 + (-k3*dldp + 2*I2*dldp/Bbar) * Y2s -4*G0*Z20/Bbar * Y2c
fxc_qsc = np.matmul(d_d_varphi,X2c) +2*iota_N*X2s + (-4*G0*Z2s/Bbar)*Y20 + 4*G0*Z20/Bbar*Y2s + (-k3*dldp + 2*I2*dldp/Bbar)*Y2c

fy0_qsc = np.matmul(d_d_varphi,Y20) + (k3*dldp - 2*I2*dldp/Bbar)*X20 -4*G0*Z2c/Bbar*X2s + 4*G0*Z2s/Bbar*X2c
fys_qsc = np.matmul(d_d_varphi,Y2s) -4*G0*Z2c/Bbar*X20 + (k3*dldp - 2*I2*dldp/Bbar)*X2s + 4*G0*Z20/Bbar*X2c -2*iota_N*Y2c
fyc_qsc = np.matmul(d_d_varphi,Y2c) + 4*G0*Z2s/Bbar*X20 -4*G0*Z20/Bbar*X2s + (k3*dldp - 2*I2*dldp/Bbar)*X2c + 2*iota_N*Y2s

eqA32_qsc = -X1s*Y2s - X1c*Y2c + X1c*Y20 + X2s*Y1s + X2c*Y1c - X20*Y1c
eqA33_qsc = Y1s*X20 - Y1c*X2s + Y1s*X2c - X1s*Y20 + X1c*Y2s - X1s*Y2c
eqA35_qsc = -X2s * qsc_stel.curvature
eqA36_qsc = -X2c * qsc_stel.curvature
eqA41_qsc = -X1s * (fx0_qsc) + X1c * (fxs_qsc) - X1s * (fxc_qsc) - Y1s * (fy0_qsc) + Y1c * (fys_qsc) - Y1s * (fyc_qsc)
eqA42_qsc = -X1c * (fx0_qsc) + X1s * (fxs_qsc) + X1c * (fxc_qsc) - Y1c * (fy0_qsc) + Y1s * (fys_qsc) + Y1c * (fyc_qsc)



print(np.max(np.abs(eqA32_qsc + eqA32_inhomogeneous_qic)))
print(np.max(np.abs(eqA33_qsc + eqA33_inhomogeneous_qic)))
print(np.max(np.abs(eqA35_qsc + eqA35_inhomogeneous_qic)))
print(np.max(np.abs(eqA36_qsc + eqA36_inhomogeneous_qic)))
print(np.max(np.abs(eqA41_qsc + eqA41_inhomogeneous_qic)))
print(np.max(np.abs(eqA42_qsc + eqA42_inhomogeneous_qic)))
print("X order 2 diff")
print(np.max(np.abs(qic_stel.X20 - qsc_stel.X20)))
print(np.max(np.abs(qic_stel.X2s - qsc_stel.X2s)))
print(np.max(np.abs(qic_stel.X2c - qsc_stel.X2c)))
print("Y order 2 diff")
print(np.max(np.abs(qic_stel.Y20 - qsc_stel.Y20)))
print(np.max(np.abs(qic_stel.Y2s - qsc_stel.Y2s)))
print(np.max(np.abs(qic_stel.Y2c - qsc_stel.Y2c)))



d_d_varphi = qsc_stel.d_d_varphi
dldp = qsc_stel.d_l_d_varphi
Z2s = qsc_stel.Z2s
Z2c = qsc_stel.Z2c
Z20 = qsc_stel.Z20
B0_over_abs_G0 = qsc_stel.B0 / np.abs(qsc_stel.G0)
B2s = qsc_stel.B2s
B2c = qsc_stel.B2c
B1c = qsc_stel.etabar
B1s = 0
X1s = qsc_stel.X1s
X1c = qsc_stel.X1c
Y1s = qsc_stel.Y1s
Y1c = qsc_stel.Y1c
k1 = qsc_stel.curvature
k2 = 0
k3 = qsc_stel.torsion
abs_G0_over_B0 = 1 / B0_over_abs_G0
iota_N = qsc_stel.iotaN
sG = qsc_stel.sG
Bbar = qsc_stel.Bbar
I2 = qsc_stel.I2
beta0 = 0
beta_1c = 0
beta_1s = qsc_stel.beta_1s

qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * k3 * dldp
qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * k3 * dldp
rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * k3 * dldp
rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * k3 * dldp


print(np.max(np.abs(qic_stel.beta_1s - qsc_stel.beta_1s)))
print(np.max(np.abs(qic_stel.beta_1c)))
print(np.max(np.abs(qic_stel.k3 - qsc_stel.torsion)))
print("X order 2")
print(np.max(np.abs(qic_stel.X20 - qsc_stel.X20)))
print(np.max(np.abs(qic_stel.X2s - qsc_stel.X2s)))
print(np.max(np.abs(qic_stel.X2c - qsc_stel.X2c)))
print("Y order 2")
print(np.max(np.abs(qic_stel.Y20 - qsc_stel.Y20)))
print(np.max(np.abs(qic_stel.Y2s - qsc_stel.Y2s)))
print(np.max(np.abs(qic_stel.Y2c - qsc_stel.Y2c)))
print("B order 2")
print(np.max(np.abs(qic_stel.B20 - qsc_stel.B20)))
print(np.max(np.abs(qic_stel.B2s - qsc_stel.B2s)))
print(np.max(np.abs(qic_stel.B2c - qsc_stel.B2c)))