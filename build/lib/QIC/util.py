import numpy as np
from scipy.interpolate import CubicSpline as spline
from QIC.centroidFrame import get_FS_frame, centroid, get_Centroid_frame, get_kappa1_kappa2, get_kappa3, get_r_vector

class Struct():
    pass



mu0 = 4 * np.pi * 1e-7


def to_Fourier(R_2D, Z_2D, nfp, mpol, ntor, lasym):
    """
    This function takes two 2D arrays (R_2D and Z_2D), which contain
    the values of the radius R and vertical coordinate Z in cylindrical
    coordinates of a given surface and Fourier transform it, outputing
    the resulting cos(theta) and sin(theta) Fourier coefficients

    The first dimension of R_2D and Z_2D should correspond to the
    theta grid, while the second dimension should correspond to the
    phi grid.

    Args:
        R_2D: 2D array of the radial coordinate R(theta, phi) of a given surface
        Z_2D: 2D array of the vertical coordinate Z(theta, phi) of a given surface
        nfp: number of field periods of the surface
        mpol: resolution in poloidal Fourier space
        ntor: resolution in toroidal Fourier space
        lasym: False if stellarator-symmetric, True if not
    """
    shape = np.array(R_2D).shape
    ntheta = shape[0]
    nphi_conversion = shape[1]
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi_conversion = np.linspace(0, 2 * np.pi / nfp, nphi_conversion, endpoint=False)
    RBC = np.zeros((int(2 * ntor + 1), int(mpol + 1)))
    RBS = np.zeros((int(2 * ntor + 1), int(mpol + 1)))
    ZBC = np.zeros((int(2 * ntor + 1), int(mpol + 1)))
    ZBS = np.zeros((int(2 * ntor + 1), int(mpol + 1)))
    factor = 2 / (ntheta * nphi_conversion)
    phi2d, theta2d = np.meshgrid(phi_conversion, theta)
    for m in range(mpol+1):
        nmin = -ntor
        if m==0: nmin = 1
        for n in range(nmin, ntor+1):
            angle = m * theta2d - n * nfp * phi2d
            sinangle = np.sin(angle)
            cosangle = np.cos(angle)
            factor2 = factor
            # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
            if np.mod(ntheta,2) == 0 and m  == (ntheta/2): factor2 = factor2 / 2
            if np.mod(nphi_conversion,2) == 0 and abs(n) == (nphi_conversion/2): factor2 = factor2 / 2
            RBC[n + ntor, m] = np.sum(R_2D * cosangle * factor2)
            RBS[n + ntor, m] = np.sum(R_2D * sinangle * factor2)
            ZBC[n + ntor, m] = np.sum(Z_2D * cosangle * factor2)
            ZBS[n + ntor, m] = np.sum(Z_2D * sinangle * factor2)
    RBC[ntor,0] = np.sum(R_2D) / (ntheta * nphi_conversion)
    ZBC[ntor,0] = np.sum(Z_2D) / (ntheta * nphi_conversion)

    if not lasym:
        RBS = 0
        ZBC = 0

    return RBC, RBS, ZBC, ZBS

def B_mag(self, r, theta, phi, Boozer_toroidal = False):
    '''
    Function to calculate the modulus of the magnetic field B for a given
    near-axis radius r, a Boozer poloidal angle theta (not vartheta) and
    a cylindrical toroidal angle phi if Boozer_toroidal = True or the
    Boozer angle varphi if Boozer_toroidal = True

    Args:
      r: the near-axis radius
      theta: the Boozer poloidal angle
      phi: the cylindrical or Boozer toroidal angle
      Boozer_toroidal: False if phi is the cylindrical toroidal angle, True for the Boozer one
    '''
    if Boozer_toroidal == False:
        thetaN = theta - (self.iota - self.iotaN) * (phi + self.nu_spline(phi))
    else:
        thetaN = theta - (self.iota - self.iotaN) * phi
    
    B1 = self.B1c_untwisted * np.cos(theta) + self.B1s_untwisted * np.sin(theta)

    B = self.B0 + r * B1
    #use scipy interpolation to extend B to nfp periods for nphi points

    # Add O(r^2) terms if necessary:
    if self.order != 'r1':
        if Boozer_toroidal == False:
            self.B20_spline = self.convert_to_spline(self.B20)
        else:
            self.B20_spline = spline(np.append(self.varphi, 2 * np.pi / self.nfp),
                                     np.append(self.B20, self.B20[0]),
                                     bc_type='periodic')

        B += (r**2) * (self.B20_spline(phi) + self.B2c * np.cos(2 * thetaN) + self.B2s * np.sin(2 * thetaN))
    
    phi_1d = np.linspace(0, 2*np.pi, self.nphi)
    phi_rollover = np.linspace(0, 2*np.pi, self.nphi+1)
    if (B.ndim == 1):
        B_rollover = np.concatenate((B, np.atleast_1d(B[0])), axis=0)
    else:
        B_rollover = np.concatenate((B, np.atleast_2d(B[:, 0]).T), axis=1)
    B_spline = spline((phi_rollover/self.nfp), B_rollover, axis=1, bc_type='periodic')
    B_full = B_spline(phi_1d)


    return B_full

def cylindrical_to_centroid(R, Z, nphi, nfp):
    # Function which calculates centroid frame from axis in cylindrical coordinates
    R0 = R[0]
    R0p = R[1]
    R0pp = R[2]
    R0ppp = R[3]
    Z0 = Z[0]
    Z0p = Z[1]
    Z0pp = Z[2]
    Z0ppp = Z[3]
    # Calculates position in cartesian coordinates
    r, rp, rpp, rppp = get_r_vector(R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp, nphi, nfp)

    # Calculates necessary diagnostics for the centroid frame from the Frenet-Serret frame
    tan, norm, binorm, curvature, torsion = get_FS_frame(rp, rpp, rppp)
    dtdp = np.empty(tan.shape)
    dldp = np.linalg.norm(rp, axis=1)
    for idx, m in enumerate(norm):
        dtdp[idx] = dldp[idx] * curvature[idx] * norm[idx]
    c = centroid(r, rp)
    p, q, dpdphi, dqdphi = get_Centroid_frame(c, r, tan, rp, dtdp)
    k1, k2 = get_kappa1_kappa2(p, q, norm, curvature)
    k3 = get_kappa3(dpdphi, dqdphi, q, p, dldp)

    return tan, norm, binorm, curvature, torsion, p, q, k1, k2 ,k3