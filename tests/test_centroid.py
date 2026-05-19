import numpy as np
from QuICGen.centroidFrame import get_FS_frame, get_Centroid_frame, get_kappa1_kappa2, centroid, get_kappa3
from QuICGen.spectral_diff_matrix import spectral_diff_matrix
import matplotlib.pyplot as plt
import unittest

class CentroidTest(unittest.TestCase):
    
    def test_centroid_frame(self):
        exCurve = np.load('./input/curve1_10xPoints.npy')
        h = 1 / len(exCurve[0])
        t, n, b, curvature, torsion = get_FS_frame(exCurve[1], exCurve[2], exCurve[3])

        #analytic
        dldp = np.linalg.norm(exCurve[1], axis=1)
        dtdp = dldp[:, None] * curvature[:, None] * n
        bprime = (np.roll(b, -1, axis=0) - np.roll(b, 1, axis=0))/(2 * h * dldp[:, None])
        torsionAlt=np.empty(torsion.shape)
        for idx, _ in enumerate(n):
            torsionAlt[idx] = -np.dot(bprime[idx],n[idx])

        atol = 0.33

        c = centroid(exCurve[0], exCurve[1])
        
        p, q, dpdphi, dqdphi = get_Centroid_frame(c, exCurve[0], t, exCurve[1], dtdp)

        pprime = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))/(2 * h * dldp[:, None])
        qprime = (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0))/(2 * h * dldp[:, None])

        np.testing.assert_allclose(dpdphi / dldp[:, None], pprime, atol = atol)
        np.testing.assert_allclose(dqdphi / dldp[:, None], qprime, atol = atol)


        k1, k2 = get_kappa1_kappa2(p, q, n, curvature)

        np.testing.assert_allclose(np.sqrt(k1**2+k2**2), curvature, atol = atol)

        k3 = get_kappa3(dpdphi, dqdphi, q, p, dldp)


        k1p = (np.roll(k1, -1) - np.roll(k1, 1))/(2 * h * dldp)
        k2p = (np.roll(k2, -1) - np.roll(k2, 1))/(2 * h * dldp)
        k3alt = np.empty(k1.shape)
        for idx in range(len(k1)):
            k3alt[idx] = torsion[idx] - (k1[idx] * k2p[idx] - k1p[idx] * k2[idx]) / (curvature[idx]*curvature[idx])


        def get_kappa1_kappa2_alt(dpdphi, dqdphi, t, lp):
        
            k1 = np.empty(t[:,0].shape)
            k2 = np.empty(t[:,0].shape)

            for idx, _ in enumerate(t):
                k1[idx] = -np.dot(dpdphi[idx], t[idx]) / lp[idx]
                k2[idx] = -np.dot(dqdphi[idx], t[idx]) / lp[idx]

            return k1, k2

        k1analytic, k2analytic = get_kappa1_kappa2_alt(dpdphi, dqdphi, t, dldp)

        np.testing.assert_allclose(k1, k1analytic, atol = atol)
        np.testing.assert_allclose(k2, k2analytic, atol = atol)
        np.testing.assert_allclose(k3, k3alt, atol = atol)


if __name__ == "__main__":
    unittest.main()