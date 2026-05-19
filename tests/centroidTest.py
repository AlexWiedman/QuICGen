import numpy as np
from QuICGen.centroidFrame import get_FS_frame, get_Centroid_frame, get_kappa1_kappa2, centroid, get_kappa3
import matplotlib.pyplot as plt
import os
from pathlib import Path

i=0
IN_DIR = (Path(__file__).parent / ".." / "tests" / "input").resolve()
curve1 = np.load(IN_DIR / 'curve1.npy')
h = 1 / len(curve1[0])

t, n, b, curvature, torsion = get_FS_frame(curve1[1], curve1[2], curve1[3])

#finite difference
dtdpFinite = (np.roll(t, -1, axis=0) - np.roll(t, 1, axis=0)) / (2 * h)
#dtdpFinite = np.empty(t.shape)

#analytic
dtdp = np.empty(t.shape)
dldp = np.linalg.norm(curve1[1], axis=1)
for idx, m in enumerate(n):
    dtdp[idx] = dldp[idx] * curvature[idx] * n[idx]
    #dtdpFinite[idx] = tprime[idx] / dldp[idx]


bprime = (np.roll(b, -1, axis=0) - np.roll(b, 1, axis=0))/(2 * h * dldp[:, None])

torsionAlt=np.empty(torsion.shape)
for idx, _ in enumerate(n):
    torsionAlt[idx] = -np.dot(bprime[idx],n[idx])

print("FS Frame")
print(t[i])
print(n[i])
print(b[i])
print(curvature[i])
print(torsion[i])


c = centroid(curve1[0], curve1[1])
print("Centroid of curve")
print(c)

p, q, dpdphi, dqdphi = get_Centroid_frame(c, curve1[0], t, curve1[1], dtdp)
print("Centroid Frame")
print(t[i])
print(p[i])
print(q[i])

print("#################")
print(dpdphi[i] / dldp[i])
print(dqdphi[i] / dldp[i])





pprime = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/(2 * h * dldp[:, None])
qprime = (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0))/(2 * h * dldp[:, None])
print(pprime[i])
print(qprime[i])
print("#################")

x = np.linspace(0,2*np.pi,len(curve1[0]))

plt.plot(x, dqdphi[:, 0] / dldp[:], label = 'analytic x component')
plt.plot(x, qprime[:, 0], label='finite difference x component')

plt.legend()
plt.show()

plt.plot(x, dqdphi[:, 1] / dldp[:], label = 'analytic y component')
plt.plot(x, qprime[:, 1], label='finite difference y component')

plt.legend()
plt.show()

plt.plot(x, dqdphi[:, 2] / dldp[:], label = 'analytic z component')
plt.plot(x, qprime[:, 2], label='finite difference z component')

plt.legend()
plt.show()




k1, k2 = get_kappa1_kappa2(p, q, n, curvature)
print("Kappa1, Kappa2")
print(k1[i])
print(k2[i])
print(np.sqrt(k1[i]**2+k2[i]**2),'=',curvature[i])


k1p = (np.roll(k1, -1) - np.roll(k1, 1))/(2 * h * dldp)
k2p = (np.roll(k2, -1) - np.roll(k2, 1))/(2 * h * dldp)
k3alt = np.empty(k1.shape)
for idx in range(len(k1)):
    k3alt[idx] = torsion[idx] - (k1[idx] * k2p[idx] - k1p[idx] * k2[idx]) / (curvature[idx]*curvature[idx])


k3 = get_kappa3(dpdphi, dqdphi, q, p, dldp)
print("Kappa 3")
print(k3[i])
print(k3alt[i])
#error was because I forgot to divide by lprime in the k3 calculation

def get_kappa1_kappa2_alt(dpdphi, dqdphi, t, lp):
    
        k1 = np.empty(t[:,0].shape)
        k2 = np.empty(t[:,0].shape)

        for idx, _ in enumerate(t):
            k1[idx] = -np.dot(dpdphi[idx], t[idx]) / lp[idx]
            k2[idx] = -np.dot(dqdphi[idx], t[idx]) / lp[idx]

        return k1, k2


k1analytic, k2analytic = get_kappa1_kappa2_alt(dpdphi, dqdphi, t, dldp)
print("Kappa 1")
print(k1analytic[i])
print(k1[i])
print("Kappa 2")
print(k2analytic[i])
print(k2[i])

plt.plot(x, k2analytic, label="Analytic")
plt.plot(x, k2, label="alternative")
plt.legend()

plt.show()


# Discrepancies disappear at higher resolution
plt.plot(x, k3, label=" Kappa 3 Analytic")
plt.plot(x, k3alt, label="Kappa 3 finite difference")
plt.legend()

plt.show()

plt.plot(x, (curvature/np.max(curvature))**2, label = 'curvature squared (normalized)')
plt.plot(x, np.abs(k3-k3alt)/np.max(k3-k3alt), label='k3 difference (normalized)')

plt.legend()
plt.show()