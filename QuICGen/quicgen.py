import numpy as np

class QuICGen():

    from QuICGen.r1calcLambda import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity_centroid, r1_diagnostics, _determine_helicity_centroid
    from .r2calc import r2calc
    from QuICGen.init_axis import init_axis, convert_to_spline
    from QuICGen.grad_B_tensor import calculate_grad_B_tensor
    from QuICGen.util import B_mag
    from QuICGen.plot import get_boundary
    from QuICGen.curve_frame_to_cylindrical import curvilinear_frame_to_cylindrical, curvilinear_frame_to_cylindrical_untwisted
    from .r_singularity import calculate_r_singularity

    def __init__(self, rc, zs, rs = None, zc = None, nfp = 1, X1c = None, Y1c = None, sigma0 = 0., B0 = None, I2 = 0
                 , sG = 1, spsi = 1, nphi = 61, B2s=None, B2c=None, p2=0., order="r1", frame="centroid"):
        
        if rs == None and zc == None:
            nfourier = np.max([len(rc), len(zs)])
        elif rs == None:
            nfourier = np.max([len(rc), len(zs), len(zc)])
        elif zc == None:
            nfourier = np.max([len(rc), len(zs), len(rs)])
        else:
            nfourier = np.max([len(rc), len(zs), len(rs), len(zc)])

        self.nfourier = nfourier
        self.rc = np.zeros(nfourier)
        self.zs = np.zeros(nfourier)
        self.rs = np.zeros(nfourier)
        self.zc = np.zeros(nfourier)
        self.rc[:len(rc)] = rc
        self.zs[:len(zs)] = zs
        self.frame = frame

        if rs != None:
            self.rs[:len(rs)] = rs
        if zc != None:
            self.zc[:len(zc)] = zc

        if np.mod(nphi, 2) == 0:
            nphi += 1

        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')

        if X1c is None:
            X1c = np.ones(nphi)
        if Y1c is None:
            Y1c = np.ones(nphi)
        if B0 is None:
            B0 = np.ones(nphi)
        elif len([B0]) == 1:
            B0 = np.full(nphi, B0)
        if B2s is None:
            B2s = np.zeros(nphi)
        if B2c is None:
            B2c = np.zeros(nphi)
        
        self.nfp = nfp
        self.sigma0 = sigma0
        self.B0 = B0
        # X1c and Y1c can't be reduced into a single unknown like etabar in the original, 
        # since the X1c' Y1c' terms are also used in a form that prevents this
        self.X1c = X1c
        self.Y1c = Y1c
        self.I2 = I2
        self.sG = sG
        self.spsi = spsi
        self.nphi = nphi
        self.B2s = B2s
        self.B2c = B2c
        self.p2 = p2
        self.order = order
        self.min_R0_threshold = 0.3

        self.calculate()
    

    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis()
        self.solve_sigma_equation()
        self.r1_diagnostics()
        if self.order != 'r1':
            self.r2calc()
            if self.order == 'r3':
                self.calculate_r3()
        self.calculate_grad_B_tensor()
