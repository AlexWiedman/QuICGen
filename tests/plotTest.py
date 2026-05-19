# Just a quick QIC test

from QICalculator_in_General_Frame.QuICGen.quicgen import QIC
import numpy as np
import logging
from QuICGen.plot import plot, B_fieldline, plot_boundary, plot_axis, B_contour
from QICCheck import test_sec_5_1, test_sec_5_2, test_sec_5_3
from qsc.qsc import Qsc

#logging.basicConfig(level=logging.DEBUG)

test, _ = test_sec_5_1()

#test = Qsc.from_paper("r1 section 5.1", nphi=63)
nphi = test.nphi


plot(test) #works fine (could use some tuning in some places though)
B_fieldline(test, nphi=nphi) #This is a little more jagged then it probably should be, not sure why
plot_boundary(test, r = 0.05, nphi=nphi, nsections=8) #works best at low aspect ratios, weird stuff when r is too large
plot_axis(test, nphi=nphi) #works fine
B_contour(test, nphi=nphi)
