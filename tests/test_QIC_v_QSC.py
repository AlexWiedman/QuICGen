"""
Compare output from QICalculator to pyQSC.
"""

import unittest
import numpy as np
import logging
from QIC.qic import QIC
from QIC.centroidFrame import get_centroid_X1_Y1
from qsc.qsc import Qsc
from qsc.plot import plot_axis
from QIC.plot import plot_axis
from QIC.to_vmec import to_vmec
#from qsc.to_vmec import to_vmec

logger = logging.getLogger(__name__)

atol = 1e-3

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


class QIC_v_QSC(unittest.TestCase):
    def test_sec_5_1(self):
        nphi = 63
        stel_qsc = Qsc.from_paper('r1 section 5.1', nphi=nphi)
        X1cfs = stel_qsc.X1c
        Y1cfs = stel_qsc.Y1c
        X1sfs = stel_qsc.X1s
        Y1sfs = stel_qsc.Y1s

        rc=stel_qsc.rc
        zs=stel_qsc.zs

        nfp = stel_qsc.nfp
        stel_qic = QIC(rc, zs, nfp = nfp, nphi=nphi)
        p = stel_qic.frame_p_cartesian
        q = stel_qic.frame_q_cartesian
        n = stel_qic.normal_cartesian
        b = stel_qic.binormal_cartesian

        X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

        stel_qic = QIC(rc, zs, B0=np.full(nphi, stel_qsc.B0), X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=0)

        np.testing.assert_allclose(X1s, stel_qic.X1s)
        np.testing.assert_allclose(Y1s, stel_qic.Y1s)
        np.testing.assert_allclose(np.mean(stel_qic.B0), stel_qsc.B0)
        np.testing.assert_allclose(np.mean(stel_qic.B1c), stel_qsc.etabar)
        np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota)
        #to_vmec(stel_qic, "input.precise_QA")

    def test_sec_5_2(self):
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
        p = stel_qic.frame_p_cartesian
        q = stel_qic.frame_q_cartesian
        n = stel_qic.normal_cartesian
        b = stel_qic.binormal_cartesian

        X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

        stel_qic = QIC(rc, zs, B0=np.full(nphi, stel_qsc.B0), X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0)

        np.testing.assert_allclose(X1s, stel_qic.X1s, atol=atol)
        np.testing.assert_allclose(Y1s, stel_qic.Y1s, atol=atol)
        np.testing.assert_allclose(np.mean(stel_qic.B0), stel_qsc.B0)
        np.testing.assert_allclose(np.mean(stel_qic.B1c), stel_qsc.etabar)
        np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota, atol=atol) 
    
    def test_precise_QA(self):
        nphi = 63
        stel_qsc = Qsc.from_paper('precise QA', nphi=nphi)
        X1cfs = stel_qsc.X1c
        Y1cfs = stel_qsc.Y1c
        X1sfs = stel_qsc.X1s
        Y1sfs = stel_qsc.Y1s

        rc=stel_qsc.rc
        zs=stel_qsc.zs

        nfp = stel_qsc.nfp
        stel_qic = QIC(rc, zs, B0=stel_qsc.B0, nfp = nfp, nphi=nphi)
        p = stel_qic.frame_p_cartesian
        q = stel_qic.frame_q_cartesian
        n = stel_qic.normal_cartesian
        b = stel_qic.binormal_cartesian

        X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

        stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, B0=stel_qsc.B0, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0)

        np.testing.assert_allclose(X1s, stel_qic.X1s)
        np.testing.assert_allclose(Y1s, stel_qic.Y1s)
        np.testing.assert_allclose(stel_qsc.curvature, stel_qic.curvature)
        np.testing.assert_allclose(np.mean(stel_qic.B0), stel_qsc.B0)
        np.testing.assert_allclose(np.mean(stel_qic.B1c) / np.min(stel_qic.B0), stel_qsc.etabar)
        np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota)
    
    def test_precise_QH(self):
        nphi = 63
        stel_qsc = Qsc.from_paper('precise QH', nphi=nphi)
        X1cfs = stel_qsc.X1c
        Y1cfs = stel_qsc.Y1c
        X1sfs = stel_qsc.X1s
        Y1sfs = stel_qsc.Y1s

        rc=stel_qsc.rc
        zs=stel_qsc.zs

        nfp = stel_qsc.nfp
        stel_qic = QIC(rc, zs, B0=stel_qsc.B0, nfp = nfp, nphi=nphi)
        p = stel_qic.frame_p_cartesian
        q = stel_qic.frame_q_cartesian
        n = stel_qic.normal_cartesian
        b = stel_qic.binormal_cartesian

        X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

        stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, B0=stel_qsc.B0, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0)

        np.testing.assert_allclose(X1s, stel_qic.X1s, atol=atol)
        np.testing.assert_allclose(Y1s, stel_qic.Y1s, atol=atol)
        np.testing.assert_allclose(stel_qsc.curvature, stel_qic.curvature)
        np.testing.assert_allclose(np.mean(stel_qic.B0), stel_qsc.B0)
        np.testing.assert_allclose(np.mean(stel_qic.B1c) / np.min(stel_qic.B0), stel_qsc.etabar)
        np.testing.assert_allclose(stel_qsc.iota, stel_qic.iota)

    def test_plunk_QI(self):
        
          

if __name__ == "__main__":
    unittest.main()