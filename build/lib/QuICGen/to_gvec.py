"""
This module contains the routines to output a
near-axis boundary to a GVEC input file
"""
from datetime import datetime
import numpy as np
from .util import mu0, to_Fourier
from scipy.interpolate import CubicSpline
import os

os.environ["OMP_NUM_THREADS"] = "2"
import gvec

def convert_RZ_to_varphi_theta_periodic(R, Z, varphi, nfp):
    """
    Convert R(phi, theta), Z(phi, theta) into R(varphi, theta), Z(varphi, theta)
    assuming periodicity in phi and varphi.

    Parameters
    ----------
    R, Z, varphi : ndarray
        Arrays of shape (ntheta, nphi)
    phi : ndarray
        1D array of phi values (periodic, e.g. 0 → 2π)

    Returns
    -------
    Rv, Zv, varphi_uniform : ndarray
        R(varphi, theta), Z(varphi, theta), and the new periodic varphi grid.
    """

    ntheta, nphi = R.shape

    # Create a uniform periodic varphi grid common to all theta
    varphi_uniform = np.linspace(0, 2*np.pi/nfp, nphi, endpoint=False)

    Rv = np.zeros_like(R)
    Zv = np.zeros_like(Z)

    for i in range(ntheta):
        varphi_i = varphi[i, :]
        R_i = R[i, :]
        Z_i = Z[i, :]

        # Sort by varphi for spline construction
        sort_idx = np.argsort(varphi_i)
        varphi_sorted = varphi_i[sort_idx]
        R_sorted = R_i[sort_idx]
        Z_sorted = Z_i[sort_idx]

        # Ensure periodic continuity
        varphi_sorted = np.concatenate([varphi_sorted, [varphi_sorted[0] + 2*np.pi/nfp]])
        R_sorted = np.concatenate([R_sorted, [R_sorted[0]]])
        Z_sorted = np.concatenate([Z_sorted, [Z_sorted[0]]])

        # Periodic cubic splines
        fR = CubicSpline(varphi_sorted, R_sorted, bc_type='periodic')
        fZ = CubicSpline(varphi_sorted, Z_sorted, bc_type='periodic')

        Rv[i, :] = fR(varphi_uniform)
        Zv[i, :] = fZ(varphi_uniform)

    return Rv, Zv, varphi_uniform

def to_gvec_RZ(self, filename, r=0.1, ntheta=20, ntorMax=14):
    params = {}
    # Project name is used in output files
    params["ProjectName"] = filename
    params["init_LA"] = True
    params["init_average_axis"] = False

    temp = - self.p2 * r * r
    am = [float(temp),-float(temp)]
    mpol1d = 100 # maximum number of mode numbers VMEC can handle
    mpol = int(np.floor(min(ntheta / 2, mpol1d)))
    ntord = 100 # maximum number of mode numbers VMEC can handle
    ntor = int(min(min(self.nphi / 2, ntord), ntorMax))
    R_2D, Z_2D, phi0_2D = self.curvilinear_frame_to_cylindrical(r, ntheta)
    
    # Fourier transform the result.
    # This is not a rate-limiting step, so for clarity of code, we don't bother with an FFT.
    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, self.nfp, mpol, ntor, self.lasym)

    # set hmap to cylinder coordindates: X1=R,X2=Z,zeta=-phi
    params["which_hmap"] = 1
    # Fourier resolution parameters: --------------------------------------------------
    params["X1_mn_max"] = (mpol, ntor)  # maximum Fourier mode numbers for X1 [m:poloidal,n:toroidal]
    params["X2_mn_max"] = (mpol, ntor)  # maximum Fourier mode numbers for X2
    params["LA_mn_max"] = (mpol, ntor)  # maximum Fourier mode numbers for LA

    # Fourier modes of axis shape -----------------------------------------------------
    params["X1_a_cos"] = {}
    # (m,n) sine Fourier mode coefficient for X2
    params["X2_a_sin"] = {}
    for n, r in enumerate(self.rc):
        params["X1_a_cos"].update({(0, n): float(r)})
    for n, z in enumerate(self.zs):
        params["X2_a_sin"].update({(0, n): float(z)})
    # Fourier modes of boundary shape -------------------------------------------------

    # number of field periods
    params["nfp"] = self.nfp
    # (m,n) cosine Fourier mode coefficient for X1
    params["X1_b_cos"] = {}
    params["X1_b_sin"] = {}
    # (m,n) sine Fourier mode coefficient for X2
    params["X2_b_sin"] = {}
    params["X2_b_cos"] = {}
    for m in range(mpol+1):
        for n in range(-ntor,ntor+1):
            if RBC[n+ntor,m]!=0 or ZBS[n+ntor,m]!=0:
                params["X1_b_cos"].update({(m, n): float(RBC[n+ntor, m])})
                params["X2_b_sin"].update({(m, n): float(ZBS[n+ntor, m])})
                if self.lasym:
                    params["X1_b_sin"].update({(m, n): float(RBS[n+ntor, m])})
                    params["X2_b_cos"].update({(m, n): float(ZBC[n+ntor, m])})
    
    # initial guess for magnetic axis is computed from the boundary shape
    params["init_average_axis"] = True

    params |= {
        # profile definitions for iota and pressure ---------------------------------------
        # total toriodal flux (scales magnetic field stength)
        "PhiEdge": np.pi * r * r * self.spsi * self.Bbar,
        "iota_type": 'polynomial',
        "iota_coefs": self.iota,

        "pres_type": 'polynomial',
        "pres_coefs": am,
        
        # radial resolution parameters ----------------------------------------------------
        # number of radial B-spline elements
        "sgrid_nElems": 2,
        # degree of B-splines for X1 and X2
        "X1X2_deg": 5,
        # degree of B-splines for LA
        "LA_deg": 5,
        # Minimizer parameters ------------------------------------------------------------
        # maximum number of iterations
        "totalIter": 10000,
        # abort tolerance on sqrt(|forces|^2) in the energy minimization
        "minimize_tol": 1e-6,
        # optional parameters -------------------------------------------------------------
        "logIter": 100,  # log interval for diagnostics
    }
    gvec.util.write_parameters(params, filename + ".ini")

def to_gvec_theta_phi(self, filename, r=0.1, ntheta=20):
    nfp = self.nfp
    nphi = self.nphi

    xyz = np.zeros((nphi*nfp, ntheta, 3))

    R_2D, Z_2D, varphi_2D = self.curvilinear_frame_to_cylindrical(r, ntheta=ntheta)

    phi1D = np.linspace(0, 2*np.pi/nfp, nphi, endpoint=False)
    R_2Dnew, Z_2Dnew, varphi_2Dnew = convert_RZ_to_varphi_theta_periodic(R_2D, Z_2D, varphi_2D, nfp)
    R_2Dnew = np.transpose(R_2Dnew, axes=(1, 0))
    Z_2Dnew = np.transpose(Z_2Dnew, axes=(1, 0))

    # X, Y, Z arrays for the whole surface
    x_2D_plot = R_2Dnew * np.cos(phi1D)[:, None]
    y_2D_plot = R_2Dnew * np.sin(phi1D)[:, None]
    z_2D_plot = Z_2Dnew

    x_2D_plot_new = x_2D_plot
    y_2D_plot_new = y_2D_plot
    z_2D_plot_new = z_2D_plot

    i = 2
    while i <= nfp:
        phi1D = np.linspace((i-1)*2*np.pi/nfp, i*2*np.pi/nfp, nphi, endpoint=False)
        x_2D_plot = R_2Dnew * np.cos(phi1D)[:, None]
        y_2D_plot = R_2Dnew * np.sin(phi1D)[:, None]
        z_2D_plot = Z_2Dnew
        x_2D_plot_new = np.append(x_2D_plot_new, x_2D_plot, axis=0)
        y_2D_plot_new = np.append(y_2D_plot_new, y_2D_plot, axis=0)
        z_2D_plot_new = np.append(z_2D_plot_new, z_2D_plot, axis=0)
        i += 1

    xyz[:,:,0] = x_2D_plot_new
    xyz[:,:,1] = y_2D_plot_new
    xyz[:,:,2] = z_2D_plot_new


    gvec.gframe.construct_gframe_from_surface(xyz, nfp, tolerance_output=1e-4, cutoff_gframe=8, filename=filename, format='toml')
    gvec.scripts.quasr.save_xyz(xyz, nfp, filename + "xyz_save.nc")