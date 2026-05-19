"""
Compare output from pyQIC (Name already taken, must rename) to pyQSC.
"""

import unittest
import numpy as np
import logging
from QuICGen.quicgen import QIC
from QuICGen.centroidFrame import get_centroid_X1_Y1
from QuICGen.util import cylindrical_to_centroid
from QuICGen.init_axis import init_axis
from qsc.qsc import Qsc

logger = logging.getLogger(__name__)

def test_sec_5_1():
        nphi = 63
        stel_qsc = Qsc.from_paper('r1 section 5.1', nphi=nphi)
        X1cfs = stel_qsc.X1c_untwisted
        Y1cfs = stel_qsc.Y1c_untwisted
        X1sfs = stel_qsc.X1s_untwisted
        Y1sfs = stel_qsc.Y1s_untwisted

        rc=stel_qsc.rc
        zs=stel_qsc.zs

        nfp = stel_qsc.nfp
        stel_qic = QIC(rc, zs, nfp = nfp, nphi=nphi)
        p = stel_qic.frame_p_cartesian
        q = stel_qic.frame_q_cartesian
        n = stel_qic.normal_cartesian
        b = stel_qic.binormal_cartesian

        X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

        stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=0, frame='centroid')

        return stel_qic, stel_qsc,
def test_sec_5_2():
        nphi = 63
        stel_qsc = Qsc.from_paper('r1 section 5.2', nphi=nphi)
        X1cfs = stel_qsc.X1c
        Y1cfs = stel_qsc.Y1c
        X1sfs = stel_qsc.X1s
        Y1sfs = stel_qsc.Y1s

        rc=stel_qsc.rc
        zs=stel_qsc.zs

        nfp = stel_qsc.nfp
        stel_qic = QIC(rc, zs, nfp = nfp, X1c=X1cfs, Y1c=Y1cfs, nphi=nphi, frame='FS')
        """p = stel_qic.frame_p_cartesian
        q = stel_qic.frame_q_cartesian
        n = stel_qic.normal_cartesian
        b = stel_qic.binormal_cartesian

        X1c, Y1c, X1s, Y1s = get_centroid_X1_Y1(p, q, n, b, [X1cfs, X1sfs], [Y1cfs, Y1sfs])

        stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0, frame='centroid')"""

        return stel_qic, stel_qsc#, X1s, Y1s

def test_sec_5_3():
        nphi = 63
        stel_qsc = Qsc.from_paper('r1 section 5.3', nphi=nphi)
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

        stel_qic = QIC(rc, zs, X1c=X1c, Y1c=Y1c, nfp = nfp, nphi = nphi, sigma0=stel_qsc.sigma0)

        return stel_qic, stel_qsc, X1s, Y1s

sqi, sqs = test_sec_5_1()

print(sqi.iotaN - sqs.iotaN)
print(sqi.iota)
print(sqs.iota)
print(sqi.iota - sqs.iota)
