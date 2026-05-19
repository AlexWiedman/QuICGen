"""
Definition set for calculating the centroid frame from the Frenet-Serret Frame, from the CoilForces.jl julia package
"""
import numpy as np

def get_r_vector(R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp, nphi, nfp):
    """
    Create a vector for the magnetic axis from R0 and Z0, with the first three derivatives.
    """

    phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    r = np.array([R0*cosphi, R0*sinphi, Z0]).transpose()
    rp = np.array([R0p*cosphi - R0*sinphi, R0p*sinphi + R0*cosphi, Z0p]).transpose()
    rpp = np.array([(R0pp - R0)*cosphi - 2*R0p*sinphi, (R0pp - R0)*sinphi + 2*R0p*cosphi, Z0pp]).transpose()
    rppp = np.array([(R0ppp - 3*R0p)*cosphi - (3*R0pp - R0)*sinphi, (R0ppp - 3*R0p)*sinphi + (3*R0pp - R0)*cosphi, Z0ppp]).transpose()


    return r, rp, rpp, rppp

def get_FS_frame(drdp, d2rdp2, d3rdp3):
    """
    Inputs: r' and r'', r''' returns Frenet-Serret Frame
    r is an n by 3 array, where n is the parametrized phi
    """
    dldp = np.linalg.norm(drdp, axis=1)

    t = np.empty(drdp.shape)
    n = np.empty(drdp.shape)
    for idx, l in enumerate (dldp):
        t[idx] = np.divide(drdp[idx], l)
        if (l * np.linalg.norm(np.cross(d2rdp2[idx], drdp[idx]))).any() == 0:
            try:
                n[idx] = (np.abs(np.divide(np.cross(drdp[idx-1,:], np.cross(d2rdp2[idx-1,:], drdp[idx-1])),(l * np.linalg.norm(np.cross(d2rdp2[idx-1], drdp[idx-1]))))) \
                       + np.abs(np.divide(np.cross(drdp[idx+1,:], np.cross(d2rdp2[idx+1,:], drdp[idx+1])),(l * np.linalg.norm(np.cross(d2rdp2[idx+1], drdp[idx+1])))))) \
                       / 2
            except:
                n[idx] = np.divide(np.cross(drdp[idx-1,:], np.cross(d2rdp2[idx-1,:], drdp[idx-1])),(l * np.linalg.norm(np.cross(d2rdp2[idx-1], drdp[idx-1])))) \
                       + np.divide(np.cross(drdp[0,:], np.cross(d2rdp2[0,:], drdp[0])),(l * np.linalg.norm(np.cross(d2rdp2[0], drdp[0]))))
        else:
            n[idx] = np.divide(np.cross(drdp[idx,:], np.cross(d2rdp2[idx,:], drdp[idx])),(l * np.linalg.norm(np.cross(d2rdp2[idx], drdp[idx]))))
        
    b = np.cross(t, n, axis=1)

    curvature = np.divide(np.linalg.norm(np.cross(drdp, d2rdp2, axis=1),axis=1),(dldp*dldp*dldp))
    torsion = np.empty(curvature.shape)
    for idx, _ in enumerate(drdp):
        if np.linalg.norm(np.cross(drdp[idx], d2rdp2[idx])) == 0:
            try:
                torsion[idx] = np.dot(drdp[idx-1], np.cross(d2rdp2[idx-1], d3rdp3[idx-1])) / np.linalg.norm(np.cross(drdp[idx-1], d2rdp2[idx-1]))**2 \
                             + np.dot(drdp[idx+1], np.cross(d2rdp2[idx+1], d3rdp3[idx+1])) / np.linalg.norm(np.cross(drdp[idx+1], d2rdp2[idx+1]))**2
            except:
                torsion[idx] = np.dot(drdp[idx-1], np.cross(d2rdp2[idx-1], d3rdp3[idx-1])) / np.linalg.norm(np.cross(drdp[idx-1], d2rdp2[idx-1]))**2 \
                             + np.dot(drdp[0], np.cross(d2rdp2[0], d3rdp3[0])) / np.linalg.norm(np.cross(drdp[0], d2rdp2[0]))**2
        
        else:
            torsion[idx] = np.dot(drdp[idx], np.cross(d2rdp2[idx], d3rdp3[idx])) / np.linalg.norm(np.cross(drdp[idx], d2rdp2[idx]))**2

    return t, n, b, curvature, torsion



def centroid(r, drdp):
    """
    Takes r and r' returns the centroid of the curve
    """
    n = len(r)

    dldp = np.linalg.norm(drdp, axis=1)

    x = np.zeros(3)
    L = 0

    for i in range(n):
        x += r[i] * dldp[i]
        L += dldp[i]

    return [0,0,0] #(1/L) * x



def get_Centroid_frame(c, r, t, drdphi, dtdphi):
    """
    Takes the centroid, r, and the tangent vector with derivatives
    Returns p and q from the centroid frame
    """
    #P0
    p = r - c
    dpdphi = drdphi.copy()
    for idx, tan in enumerate(t):
        #P1
        temp = np.dot(p[idx], tan)
        dtempdphi = np.dot(p[idx], dtdphi[idx]) + np.dot(dpdphi[idx], tan)

        p[idx] -= temp * tan
        dpdphi[idx] = dpdphi[idx] - dtempdphi * tan - temp * dtdphi[idx]

        #Pf
        temp2 = np.linalg.norm(p[idx])
        p[idx] = (1 / temp2) * p[idx]
        dpdphi[idx] =  (1 / temp2) * dpdphi[idx] - 1/(temp2) * p[idx] * np.dot(dpdphi[idx], p[idx])

        

    return p, np.cross(t, p, axis=1), dpdphi, (np.cross(t, dpdphi, axis=1) + np.cross(dtdphi, p, axis=1)) 



def get_kappa1_kappa2(p, q, n, curvature):
    """
    Takes p and q from a centroid frame, n from a Frenet-Serret frame, and the curvature
    Returns the curvatures kappa1 and kappa2
    """
    k1 = np.empty(n[:,0].shape)
    k2 = np.empty(n[:,0].shape)
    for idx, _ in enumerate(n):
        cos_a = np.dot(p[idx], n[idx])
        minus_sin_a = np.dot(q[idx], n[idx])
        k1[idx] = curvature[idx] * cos_a
        k2[idx] = curvature[idx] * minus_sin_a

    return k1, k2

def get_centroid_X1_Y1(p, q, n, b, X1fs, Y1fs):
    """
    Calculates X1 and Y1 in the centroid frame from the p and q vectors from centroid frame, as well as the n and b vectors and X1 and Y1 in the Frenet-Serret frame
    """
    X1c = np.empty(len(X1fs[0]))
    X1s = np.empty(len(X1fs[1]))
    Y1c = np.empty(len(Y1fs[0]))
    Y1s = np.empty(len(Y1fs[1]))

    for idx, _ in enumerate(n):
        ndotp = np.dot(n[idx], p[idx])
        bdotp = np.dot(b[idx], p[idx])
        ndotq = np.dot(n[idx], q[idx])
        bdotq = np.dot(b[idx], q[idx])
        X1c[idx] = X1fs[0][idx] * ndotp + Y1fs[0][idx] * bdotp
        X1s[idx] = (X1fs[1][idx] * ndotp + Y1fs[1][idx] * bdotp)
        Y1c[idx] = X1fs[0][idx] * ndotq + Y1fs[0][idx] * bdotq
        Y1s[idx] = (X1fs[1][idx] * ndotq + Y1fs[1][idx] * bdotq)

    return X1c, Y1c, X1s, Y1s

def get_centroid_X2_Y2(p, q, n, b, X2fs, Y2fs):
    """
    Input p and q vectors from the centroid frame, the n and b vectors and X1 and Y1 in the Frenet-Serret frame
    Returns X2 and Y2 in the centroid frame
    """
    X20 = np.empty(len(X2fs[0]))
    X2c = np.empty(len(X2fs[1]))
    X2s = np.empty(len(X2fs[2]))
    Y20 = np.empty(len(X2fs[0]))
    Y2c = np.empty(len(Y2fs[1]))
    Y2s = np.empty(len(Y2fs[2]))

    for idx, _ in enumerate(n):
        ndotp = np.dot(n[idx], p[idx])
        bdotp = np.dot(b[idx], p[idx])
        ndotq = np.dot(n[idx], q[idx])
        bdotq = np.dot(b[idx], q[idx])
        X20[idx] = X2fs[0][idx] * ndotp + Y2fs[0][idx] * bdotp
        X2c[idx] = X2fs[1][idx] * ndotp + Y2fs[1][idx] * bdotp
        X2s[idx] = X2fs[2][idx] * ndotp + Y2fs[2][idx] * bdotp

        Y20[idx] = X2fs[0][idx] * ndotq + Y2fs[0][idx] * bdotq
        Y2c[idx] = X2fs[1][idx] * ndotq + Y2fs[1][idx] * bdotq
        Y2s[idx] = X2fs[2][idx] * ndotq + Y2fs[2][idx] * bdotq

    return X20, Y20, X2c, Y2c, X2s, Y2s

def get_kappa3(dpdphi, dqdphi, q, p, lp, tol=1e-10):
    """
    Input q and p from centroid frame, and first derivatives
    Output kappa 3
    """

    k3_0 = np.empty(p[:,0].shape)
    k3_1 = np.empty(p[:,0].shape)
    k3 = np.empty(p[:,0].shape)
    for idx, _ in enumerate(p):
        k3_0[idx] = np.dot(dpdphi[idx], q[idx]) / lp[idx]
        k3_1[idx] = -np.dot(dqdphi[idx], p[idx]) / lp[idx]
        k3[idx] = (k3_0[idx]+k3_1[idx]) / 2

    if np.average(np.abs(k3_0-k3_1)) > tol:
        print("Distance between methods is larger than tolerance")

    return k3

