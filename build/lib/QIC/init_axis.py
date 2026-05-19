import numpy as np
from scipy.interpolate import CubicSpline as spline
from QIC.spectral_diff_matrix import spectral_diff_matrix
from QIC.centroidFrame import get_FS_frame, centroid, get_Centroid_frame, get_kappa1_kappa2, get_kappa3, get_r_vector
"""
Calculate the magnetic axis 
"""

def convert_to_spline(self,array):
    sp=spline(np.append(self.phi,2*np.pi/self.nfp), np.append(array,array[0]), bc_type='periodic')
    return sp

def init_axis(self):
    nphi = self.nphi
    nfp = self.nfp

    phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    d_phi = phi[1] - phi[0]
    R0 = np.zeros(nphi)
    Z0 = np.zeros(nphi)
    R0p = np.zeros(nphi)
    Z0p = np.zeros(nphi)
    R0pp = np.zeros(nphi)
    Z0pp = np.zeros(nphi)
    R0ppp = np.zeros(nphi)
    Z0ppp = np.zeros(nphi)
    # Calculation of R0 and Z0 with first three derivatives from fourier series
    for jn in range(0, self.nfourier):
        n = jn * nfp
        sinangle = np.sin(n * phi)
        cosangle = np.cos(n * phi)
        R0 += self.rc[jn] * cosangle + self.rs[jn] * sinangle
        Z0 += self.zc[jn] * cosangle + self.zs[jn] * sinangle
        R0p += self.rc[jn] * (-n * sinangle) + self.rs[jn] * (n * cosangle)
        Z0p += self.zc[jn] * (-n * sinangle) + self.zs[jn] * (n * cosangle)
        R0pp += self.rc[jn] * (-n * n * cosangle) + self.rs[jn] * (-n * n * sinangle)
        Z0pp += self.zc[jn] * (-n * n * cosangle) + self.zs[jn] * (-n * n * sinangle)
        R0ppp += self.rc[jn] * (n * n * n * sinangle) + self.rs[jn] * (-n * n * n * cosangle)
        Z0ppp += self.zc[jn] * (n * n * n * sinangle) + self.zs[jn] * (-n * n * n * cosangle)
    
    d_l_d_phi = np.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
    d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
    self.d_d_phi = spectral_diff_matrix(self.nphi, xmax=2 * np.pi / self.nfp)

    ### Use the G0 calculation from https://github.com/rogeriojorge/pyQIC/
    # Calculates G0 and varphi by solving for nu from the definition of varphi
    nu = np.zeros((nphi,))
    for j in range(20):
        varphi = phi + nu
        abs_G0 = np.sum(self.B0 * d_l_d_phi) / nphi
        rhs = -1 + d_l_d_phi * self.B0 / abs_G0
        last_nu = nu
        nu = np.linalg.solve(self.d_d_phi, rhs)
        norm_change = np.sqrt(sum((nu-last_nu)**2)/nphi)
        if norm_change < 1e-13:
            break
    varphi = phi + nu - nu[0]
    self.nu = nu
    self.varphi = varphi
    self.d_varphi_d_phi = 1 + np.matmul(self.d_d_phi, nu - nu[0])
    
    d_l_d_varphi = d_l_d_phi / self.d_varphi_d_phi

    G0 = self.sG * np.sum(self.B0 * d_l_d_varphi) / nphi

    # For these next arrays, the first dimension is phi, and the 2nd dimension is (R, phi, Z).
    d_r_d_phi_cylindrical = np.array([R0p, R0, Z0p]).transpose()
    d2_r_d_phi2_cylindrical = np.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
    d3_r_d_phi3_cylindrical = np.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()
        
    # Calculates position in cartesian coordinates
    r, rp, rpp, rppp = get_r_vector(R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp, nphi, nfp)
    # Calculates necessary diagnostics for the centroid frame from the Frenet-Serret frame
    tangent, normal, binormal, curvature, torsion = get_FS_frame(rp, rpp, rppp)

    if self.frame == "centroid":
        dtdp = np.empty(tangent.shape)
        dldp = np.linalg.norm(rp, axis=1)
        for idx, m in enumerate(normal):
            dtdp[idx] = dldp[idx] * curvature[idx] * normal[idx]
        c = centroid(r, rp)
        p, q, dpdphi, dqdphi = get_Centroid_frame(c, r, tangent, rp, dtdp)

        k1, k2 = get_kappa1_kappa2(p, q, normal, curvature)
        k3 = get_kappa3(dpdphi, dqdphi, q, p, dldp)
    elif self.frame == "FS":
        k1 = curvature
        k2 = np.zeros(k1.shape)
        k3 = torsion
        p = normal
        q = binormal


    
    self.normal_cartesian = normal
    self.binormal_cartesian = binormal
    self._determine_helicity_centroid()

    axis_length = np.sum(d_l_d_phi) * d_phi * nfp
    
    self.d_d_varphi = np.zeros((nphi, nphi))
    for j in range(nphi):
        self.d_d_varphi[j,:] = self.d_d_phi[j,:] / self.d_varphi_d_phi[j] 


    # Add all results to self:
    self.phi = phi
    self.d_phi = d_phi
    self.R0 = R0
    self.Z0 = Z0
    self.R0p = R0p
    self.Z0p = Z0p
    self.R0pp = R0pp
    self.Z0pp = Z0pp
    self.R0ppp = R0ppp
    self.Z0ppp = Z0ppp
    self.G0 = G0
    self.d_l_d_phi = d_l_d_phi
    self.axis_length = axis_length
    self.curvature = curvature
    self.torsion = torsion
    self.k1 = k1
    self.k2 = k2
    self.k3 = k3

    np.testing.assert_allclose(np.sqrt(k1**2+k2**2), curvature, atol=1e-15)

    self.tangent_cartesian = tangent
    self.frame_p_cartesian = p
    self.frame_q_cartesian = q

    # Save Centroid frame in cylindrical coordinates
    tan_R = np.zeros(nphi)
    tan_phi = np.zeros(nphi)
    tan_z = np.zeros(nphi)

    p_R = np.zeros(nphi)
    p_phi = np.zeros(nphi)
    p_z = np.zeros(nphi)

    q_R = np.zeros(nphi)
    q_phi = np.zeros(nphi)
    q_z = np.zeros(nphi)

    for phiIndex in range(nphi):
        phi0 = phi[phiIndex]
        sinphi0 = np.sin(phi0)
        cosphi0 = np.cos(phi0)
        
        p_R[phiIndex] = p[phiIndex,0] * cosphi0 + p[phiIndex,1] * sinphi0
        p_phi[phiIndex] = -p[phiIndex,0] * sinphi0 + p[phiIndex,1] * cosphi0
        p_z[phiIndex] = p[phiIndex,2]

        q_R[phiIndex] = q[phiIndex,0] * cosphi0 + q[phiIndex,1] * sinphi0
        q_phi[phiIndex] = -q[phiIndex,0] * sinphi0 + q[phiIndex,1] * cosphi0
        q_z[phiIndex] = q[phiIndex,2]

        tan_R[phiIndex] = tangent[phiIndex,0] * cosphi0 + tangent[phiIndex,1] * sinphi0
        tan_phi[phiIndex] = -tangent[phiIndex,0] * sinphi0 + tangent[phiIndex,1] * cosphi0
        tan_z[phiIndex] = tangent[phiIndex,2]


    self.tangent_cylindrical = np.array([tan_R, tan_phi, tan_z]).T
    self.frame_p_cylindrical =  np.array([p_R, p_phi, p_z]).T
    self.frame_q_cylindrical = np.array([q_R, q_phi, q_z]).T
    self.Bbar = self.spsi * np.mean(self.B0)
    self.abs_G0_over_B0 = d_l_d_varphi
    self.d_l_d_varphi = d_l_d_varphi

    self.lasym = np.max(np.abs(self.rs)) > 0 or np.max(np.abs(self.zc)) > 0 \
        or self.sigma0 != 0

    # Functions that converts a toroidal angle phi0 on the axis to the axis radial and vertical coordinates
    self.R0_func = self.convert_to_spline(sum([self.rc[i]*np.cos(i*self.nfp*self.phi) +\
                                               self.rs[i]*np.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.rc))]))
    self.Z0_func = self.convert_to_spline(sum([self.zc[i]*np.cos(i*self.nfp*self.phi) +\
                                               self.zs[i]*np.sin(i*self.nfp*self.phi) \
                                              for i in range(len(self.zs))]))
    
    # Spline interpolants for the cartesian components of the Centroid frame:
    self.frame_p_R_spline         = self.convert_to_spline(self.frame_p_cylindrical[:,0])
    self.frame_p_phi_spline       = self.convert_to_spline(self.frame_p_cylindrical[:,1])
    self.frame_p_z_spline         = self.convert_to_spline(self.frame_p_cylindrical[:,2])
    self.frame_q_R_spline         = self.convert_to_spline(self.frame_q_cylindrical[:,0])
    self.frame_q_phi_spline       = self.convert_to_spline(self.frame_q_cylindrical[:,1])
    self.frame_q_z_spline         = self.convert_to_spline(self.frame_q_cylindrical[:,2])
    self.tangent_R_spline         = self.convert_to_spline(self.tangent_cylindrical[:,0])
    self.tangent_phi_spline       = self.convert_to_spline(self.tangent_cylindrical[:,1])
    self.tangent_z_spline         = self.convert_to_spline(self.tangent_cylindrical[:,2])

    # Spline interpolant for nu = varphi - phi, used for plotting
    self.nu_spline = self.convert_to_spline(self.varphi - self.phi)