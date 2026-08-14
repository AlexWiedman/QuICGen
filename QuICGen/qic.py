import numpy as np

class QIC():

    from QIC.r1calcLambda import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity_centroid, r1_diagnostics
    from .r2calc import r2calc
    from QIC.init_axis import init_axis, convert_to_spline
    from QIC.grad_B_tensor import calculate_grad_B_tensor
    from QIC.util import B_mag
    from QIC.plot import get_boundary, plot_boundary
    from QIC.curve_frame_to_cylindrical import curvilinear_frame_to_cylindrical, curvilinear_frame_to_cylindrical_untwisted
    from .r_singularity import calculate_r_singularity
    from QIC.r1calcQIC import FS_residual, FS_jacobian, FS_determine_helicity, FS_solve_sigma_equation, FS_r1_diagnostics, convert_qic_results
    from QIC.init_axis_general import init_axis_general, convert_to_spline

    def __init__(self, rc, zs, rs = None, zc = None, nfp = 1, X1c = None, Y1c = None, sigma0 = 0., B0 = None, I2 = 0
                 , sG = 1, spsi = 1, nphi = 61, B2s=None, B2c=None, p2=0., order="r1", frame="centroid"
                 , omn = False, omn_method = 'buffer', delta = np.pi/5, d_over_curvature=0, d_over_curvature_cvals = []
                 , B0_vals=[], B0_svals=[], d_cvals=[], d_svals=[], alpha_cvals=[0.], alpha_svals=[0.], phi_shift=0
                 , etabar = 1., input = 'FS', k_second_order_SS = 0):
        
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
        if input == 'FS':
        

            #added new variables to initialize
            self.omn = omn
            self.omn_method = omn_method
            self.delta  = min(abs(delta),0.95*np.pi/nfp)
            self.d_over_curvature = d_over_curvature
            self.d_over_curvature_cvals = d_over_curvature_cvals
            self.B0_vals = B0_vals
            self.B0_svals = B0_svals
            self.d_cvals = d_cvals
            self.d_svals = d_svals
            self.alpha_cvals = alpha_cvals
            self.alpha_svals = alpha_svals
            self.phi_shift = phi_shift
            self.etabar = etabar
            self.input = input
            self.k_second_order_SS = k_second_order_SS

            #added new methods to calculate pyQIC values
            phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
            self.d_phi = phi[1] - phi[0]
            if omn==True and phi_shift==0:
                phi_shift=1/3
            self.phi_shift = phi_shift
            self.phi = phi + self.phi_shift*self.d_phi
            if B0_vals==[]:
                self.B0_vals = [B0]
                self.B0_well_depth = 0
            else:
                self.B0_vals = B0_vals
                self.B0_well_depth = B0_vals[1]
            self.B0 =  np.array(sum([self.B0_vals[i]*np.cos(nfp*i*self.phi) for i in range(len(self.B0_vals))]))
            self.B0_svals = B0_svals
            self.B0 += np.array(sum([self.B0_svals[i]*np.sin(nfp*i*self.phi) for i in range(len(self.B0_svals))]))
            #print(B0)
            if d_cvals==[] and d_svals == [] and d_over_curvature_cvals == [] and d_over_curvature == 0:
                self.d_cvals = [etabar]
                self.d_svals = d_svals
            else:
                self.d_cvals = d_cvals
                self.d_svals = d_svals
                if self.d_cvals == []:
                    self.etabar = 0
                    self.d_cvals = [0]
                else:
                    self.etabar = d_cvals[0]
            if d_over_curvature_cvals == [] and not d_over_curvature == 0:
                self.d_over_curvature_cvals == [d_over_curvature]
            if not d_over_curvature_cvals == [] and not d_over_curvature == 0:
                raise ValueError('Only one of d_over_curvature_cvals or d_over_curvature should be specified.')
            self.d = np.array(sum([self.d_cvals[i]*np.cos(nfp*i*self.phi) for i in range(len(self.d_cvals))]))
            self.d = self.d + np.array(sum([self.d_svals[i]*np.sin(nfp*i*self.phi) for i in range(len(self.d_svals))]))
            self.alpha = np.array(sum([self.alpha_cvals[i]*np.cos(nfp*i*self.phi) for i in range(len(self.alpha_cvals))]))
            self.alpha = self.alpha + np.array(sum([self.alpha_svals[i]*np.sin(nfp*i*self.phi) for i in range(len(self.alpha_svals))]))
            self.B1s = self.B0 * self.d * np.sin(self.alpha)
            self.B1c = self.B0 * self.d * np.cos(self.alpha)

        self.calculate()
    

    def calculate(self):
        """
        Driver for the main calculations.
        """
        
        if input == 'not FS':
            print("running")
            self.init_axis_general()
        else:
            self.init_axis()
            self.FS_solve_sigma_equation()
            self.FS_r1_diagnostics()
            self.convert_qic_results()
        self.solve_sigma_equation()
        self.r1_diagnostics()
        
        if self.order != 'r1':
            self.r2calc()
            if self.order == 'r3':
                self.calculate_r3()
        self.calculate_grad_B_tensor()