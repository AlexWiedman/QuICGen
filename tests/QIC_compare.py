"""
Compare output from pyQIC (Name already taken, must rename) to pyQSC.
"""

import unittest
import numpy as np
import logging
from QICalculator_in_General_Frame.QuICGen.quicgen import QIC
from QuICGen.centroidFrame import get_centroid_X1_Y1
from QuICGen.util import cylindrical_to_centroid
from QuICGen.init_axis import init_axis
from qsc.qsc import Qsc

logger = logging.getLogger(__name__)

def compare_to_fortran_r1(filename):
    filler = np.load(filename) # placeholder for the time being, until I actually get a file to compare to

    dat = filler.x
    nphi = dat.nphi
    nfp = dat.nfp
    QSx1c = dat.x1c
    QSx1s = dat.x1s
    QSy1c = dat.y1c
    QSy1s = dat.y1s
    QSB0 = np.full(nphi, dat.B0)
    QSB1s = np.full(nphi, dat.B1s)
    QSB1c = np.full(nphi, dat.B1c)
    QSrc = dat.rc
    QSzs = dat.zs
    sigma0 = dat.sigma0
    sG = dat.sG
    spsi = dat.spsi

    def compare_field(qsc_field, qic_field, rtol=1e-9, atol=1e-9):
        logger('max difference: {}'.format(np.max(np.abs(qsc_field - qic_field))))
        np.testing.assert_allclose(qsc_field, qic_field, rtol=rtol, atol=atol)

    iso = QIC(QSrc, QSzs, nfp = nfp, sigma0 = sigma0, X1c = QSx1c, Y1c=QSy1c, B0=QSB0, nphi=nphi, sG=sG, spsi=spsi)
    iso.calculate()

    compare_field(QSx1s, iso.X1s)
    compare_field(QSy1s, iso.y1s)
    compare_field(QSB1s, iso.B1s)
    compare_field(QSB1c, iso.B1c)
    compare_field(dat.iota, iso.iota)
    compare_field(dat.torsion, iso.torsion)
    compare_field(dat.curvature, iso.curvature)
    compare_field(dat.sigma, iso.sigma)


    
def sec_5_1():
    Y1cfs = [-0, 0.05474354,  0.10944288,  0.1640492,   0.2185044,   0.27273643, \
  0.32665441,  0.38014374,  0.43306088,  0.48522797,  0.53642708,  0.58639394, \
  0.63481105,  0.68130002,  0.72541255,  0.76661993,  0.80430033,  0.83772321, \
  0.86603009,  0.88821088,  0.90307536,  0.90922039,  0.90499591,  0.88847777, \
  0.85746461,  0.80953157,  0.74219453,  0.65325826,  0.54141501,  0.40708105, \
  0.2532666,   0.08602639, -0.08602639, -0.2532666,  -0.40708105, -0.54141501, \
 -0.65325826, -0.74219453, -0.80953157, -0.85746461, -0.88847777, -0.90499591, \
 -0.90922039, -0.90307536, -0.88821088, -0.86603009, -0.83772321, -0.80430033, \
 -0.76661993, -0.72541255, -0.68130002, -0.63481105, -0.58639394, -0.53642708, \
 -0.48522797, -0.43306088, -0.38014374, -0.32665441, -0.27273643, -0.2185044, \
 -0.1640492,  -0.10944288, -0.05474354]
    X1cfs = [-0.68912069, -0.68953755, -0.69079464, -0.69291161, -0.69592151, -0.69987127, \
 -0.70482237, -0.71085173, -0.7180529,  -0.7265374,  -0.73643644, -0.74790289, \
 -0.76111349, -0.77627136, -0.79360851, -0.81338828, -0.83590715, -0.8614952, \
 -0.89051383, -0.92334862, -0.96039382, -1.00202286, -1.04853682, -1.10007921, \
 -1.1565031,  -1.21717748, -1.28073195, -1.34477339, -1.40567565, -1.45863345, \
 -1.49821301, -1.51949172, -1.51949172, -1.49821301, -1.45863345, -1.40567565, \
 -1.34477339, -1.28073195, -1.21717748, -1.1565031,  -1.10007921, -1.04853682, \
 -1.00202286, -0.96039382, -0.92334862, -0.89051383, -0.8614952,  -0.83590715, \
 -0.81338828, -0.79360851, -0.77627136, -0.76111349, -0.74790289, -0.73643644, \
 -0.7265374,  -0.7180529,  -0.71085173, -0.70482237, -0.69987127, -0.69592151, \
 -0.69291161, -0.69079464, -0.68953755]
    Y1sfs = [-1.45112462, -1.45024735, -1.44760822, -1.44318551, -1.43694365, -1.42883419,
 -1.4187972,  -1.40676312, -1.39265505, -1.37639163, -1.35789043, -1.33707199,
 -1.3138645,  -1.28820932, -1.26006714, -1.22942515, -1.19630511, -1.16077257,
 -1.12294719, -1.08301456, -1.04123952, -0.99798122, -0.95370995, -0.90902545,
 -0.86467559, -0.82157288, -0.78080351, -0.74361971, -0.71140167, -0.6855732,
 -0.66746183, -0.65811481, -0.65811481, -0.66746183, -0.6855732,  -0.71140167,
 -0.74361971, -0.78080351, -0.82157288, -0.86467559, -0.90902545, -0.95370995,
 -0.99798122, -1.04123952, -1.08301456, -1.12294719, -1.16077257, -1.19630511,
 -1.22942515, -1.26006714, -1.28820932, -1.3138645,  -1.33707199, -1.35789043,
 -1.37639163, -1.39265505, -1.40676312, -1.4187972,  -1.42883419, -1.43694365,
 -1.44318551, -1.44760822, -1.45024735]
    X1sfs = np.zeros(len(Y1sfs))

    rc=[1, 0.045]
    zs=[0, -0.045]

    
    nphi = 63
    nfp=3
    stel = QIC(rc, zs, nfp = nfp, nphi=nphi)
    p = stel.frame_p
    q = stel.frame_q
    n = stel.normal
    b = stel.binormal

    X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])
    #np.testing.assert_allclose(X1c, X1cfs, atol=1e-15)
    #np.testing.assert_allclose(Y1c, Y1cfs, atol=1e-15)
    #np.testing.assert_allclose(X1s, X1sfs, atol=1e-15)
    #np.testing.assert_allclose(Y1s, Y1sfs, atol=1e-15)
    #np.testing.assert_allclose()

    stel = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=0)

    """
    X1cP = np.matmul(stel.d_d_varphi, stel.X1c)
    Y1cP = np.matmul(stel.d_d_varphi, stel.Y1c)
    B0P = np.matmul(stel.d_d_varphi, stel.B0)
    L2 = np.power(stel.X1c,2) + np.power(stel.Y1c,2)
    beta = -1 * stel.Bbar / (stel.B0 * L2)

    r = np.matmul(stel.d_d_varphi, stel.sigma) - stel.sigma * (B0P / stel.B0 + 2 * (stel.X1c * X1cP + stel.Y1c * Y1cP) / L2) \
          + stel.iota * beta * (1 + 1 / (beta * beta) + stel.sigma * stel.sigma) \
          + 2 * stel.sG * (stel.Y1c * X1cP - stel.X1c * Y1cP) / L2 \
          + 2 * (stel.I2 / stel.Bbar - stel.k3) * stel.G0 / stel.B0
    """
    
    #print(np.average(r))
    #print(stel.iota)
    #print("X1s diff")
    #print(np.abs(stel.X1s - X1s))
    #print("Y1s diff")
    #print(np.abs(stel.Y1s - Y1s))
    #print(np.average(stel.X1s))
    #print(np.average(stel.Y1s))
    #print("B1c")
    #print(stel.B1c / stel.B0)
    #print("B1s")
    #print(stel.B1s / stel.B0)
    #print(stel.k1)
    #print(stel.k2)
    #print(np.sqrt(stel.k1**2+stel.k2**2))
    #print(stel.curvature)
    #print(stel.frame_p[0])
    #print(stel.frame_q[0])
    #print(stel.tangent[0])
    
    return stel

nphi = 63
stel_qsc = Qsc.from_paper('r1 section 5.2', nphi=nphi)
X1cfs = stel_qsc.X1c
Y1cfs = stel_qsc.Y1c
X1sfs = stel_qsc.X1s
Y1sfs = stel_qsc.Y1s

rc=stel_qsc.rc
zs=stel_qsc.zs

nfp = stel_qsc.nfp
stel_qic = QIC(rc, zs, nfp = nfp, nphi=nphi)
p = stel_qic.frame_p
q = stel_qic.frame_q
n = stel_qic.normal
b = stel_qic.binormal

X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0)

print(stel_qic.iota)

np.testing.assert_allclose(X1s, stel_qic.X1s)
np.testing.assert_allclose(Y1s, stel_qic.Y1s)
np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota)