"""
Definition set for calculating the centroid frame from the Frenet-Serret Frame, from the CoilForces.jl julia package
"""
import numpy as np
def get_r_vector(R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp, nphi, nfp):

    phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    r = np.array([R0*cosphi, R0*sinphi, Z0]).transpose()
    rp = np.array([R0p*cosphi - R0*sinphi, R0p*sinphi + R0*cosphi, Z0p]).transpose()
    rpp = np.array([(R0pp - R0)*cosphi - 2*R0p*sinphi, (R0pp - R0)*sinphi + 2*R0p*cosphi, Z0pp]).transpose()
    rppp = np.array([(R0ppp - 3*R0p)*cosphi - (3*R0pp - R0)*sinphi, (R0ppp - 3*R0p)*sinphi + (3*R0pp - R0)*cosphi, Z0ppp]).transpose()


    return r, rp, rpp, rppp

def get_r_vector(R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp, nphi, nfp):

    phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    r = np.array([R0*cosphi, R0*sinphi, Z0]).transpose()
    rp = np.array([R0p*cosphi - R0*sinphi, R0p*sinphi + R0*cosphi, Z0p]).transpose()
    rpp = np.array([(R0pp - R0)*cosphi - 2*R0p*sinphi, (R0pp - R0)*sinphi + 2*R0p*cosphi, Z0pp]).transpose()
    rppp = np.array([(R0ppp - 3*R0p)*cosphi - (3*R0pp - R0)*sinphi, (R0ppp - 3*R0p)*sinphi + (3*R0pp - R0)*cosphi, Z0ppp]).transpose()


    return r, rp, rpp, rppp

def get_FS_frame(drdp, d2rdp2, d3rdp3):
    # takes r' and r'', r''' returns Frenet-Serret Frame
    # r is an n by 3 array, where n is the parametrized phi
    
    dldp = np.linalg.norm(drdp, axis=1)

    t = np.empty(drdp.shape)
    n = np.empty(drdp.shape)
    for idx, l in enumerate (dldp):
        t[idx] = np.divide(drdp[idx], l)
        n[idx] = np.divide(np.cross(drdp[idx,:], np.cross(d2rdp2[idx,:], drdp[idx])),(l * np.linalg.norm(np.cross(d2rdp2[idx], drdp[idx]))))
    b = np.cross(t, n, axis=1)

    curvature = np.divide(np.linalg.norm(np.cross(drdp, d2rdp2, axis=1),axis=1),(dldp*dldp*dldp))
    torsion = np.empty(curvature.shape)
    for idx, _ in enumerate(drdp):
        torsion[idx] = np.dot(drdp[idx], np.cross(d2rdp2[idx], d3rdp3[idx])) / np.linalg.norm(np.cross(drdp[idx], d2rdp2[idx]))**2

    return t, n, b, curvature, torsion

def get_kappa1_kappa2(curvature):
    # Unnecessary in FS frame, but kept for consistency with general frame

    k1 = curvature
    k2 = 0

    return k1, k2

def get_kappa3(torsion):

    k3 = torsion
    
    return k3