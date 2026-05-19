#!/usr/bin/env python3

"""
Functions for computing the maximum r at which the flux surfaces
become singular.
"""

import logging
import warnings
import numpy as np
#from .util import Struct, fourier_minimum

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_r_singularity(self, high_order=False):
    """
    Find the minimum r where a second order configuration would have a self intersecting boundary
    """

    # Shorthand:
    s = self
    
    X1s = s.X1s
    X1c = s.X1c
    Y1s = s.Y1s
    Y1c = s.Y1c

    X20 = s.X20
    X2s = s.X2s
    X2c = s.X2c

    Y20 = s.Y20
    Y2s = s.Y2s
    Y2c = s.Y2c

    Z20 = s.Z20
    Z2s = s.Z2s
    Z2c = s.Z2c

    iota_N0 = s.iotaN
    iota = s.iota
    lp = np.abs(s.G0) / s.B0

    k1 = s.k1
    k2 = s.k2
    k3 = s.k3

    nphi = s.nphi
    sign_G = s.sG
    sign_psi = s.spsi
    B0 = s.B0
    G0 = s.G0
    I2 = s.I2
    G2 = s.G2
    p2 = s.p2

    B20 = s.B20
    B2s = s.B2s
    B2c = s.B2c

    d_X1s_d_varphi = s.d_X1s_d_varphi
    d_X1c_d_varphi = s.d_X1c_d_varphi
    d_Y1s_d_varphi = s.d_Y1s_d_varphi
    d_Y1c_d_varphi = s.d_Y1c_d_varphi

    d_X20_d_varphi = s.d_X20_d_varphi
    d_X2s_d_varphi = s.d_X2s_d_varphi
    d_X2c_d_varphi = s.d_X2c_d_varphi

    d_Y20_d_varphi = s.d_Y20_d_varphi
    d_Y2s_d_varphi = s.d_Y2s_d_varphi
    d_Y2c_d_varphi = s.d_Y2c_d_varphi

    d_Z20_d_varphi = s.d_Z20_d_varphi
    d_Z2s_d_varphi = s.d_Z2s_d_varphi
    d_Z2c_d_varphi = s.d_Z2c_d_varphi

    d2_X1c_d_varphi2 = s.d2_X1c_d_varphi2
    d2_Y1s_d_varphi2 = s.d2_Y1s_d_varphi2
    d2_Y1c_d_varphi2 = s.d2_Y1c_d_varphi2

    r_singularity_basic_vs_varphi = np.zeros(nphi)
    r_singularity_vs_varphi = np.zeros(nphi)
    r_singularity_residual_sqnorm = np.zeros(nphi)
    r_singularity_theta_vs_varphi = np.zeros(nphi)
    
    # Write sqrt(g) = r * [g0 + r*g1c*cos(theta) + (r^2)*(g20 + g2s*sin(2*theta) + g2c*cos(2*theta) + ...]
    # The coefficients are evaluated in "20200322-02 Max r for Garren Boozer.nb", in the section "Order r^2 construction, quasisymmetry"

    g0 = lp * (X1c * Y1s - X1s * Y1c)

    g1s = lp*(-2*X20*Y1c+2*X2c*Y1c+2*X2s*Y1s+2*X1c*Y20-2*X1c*Y2c-2*X1s*Y2s+X1s*X1s*Y1c*k1-X1c*X1s*Y1s*k1+X1s*Y1c*Y1s*k2-X1c*Y1s*Y1s*k2)
    # g1s vanishes for quasisymmetry.

    g1c = lp*(-2*X2s*Y1c + 2*X20*Y1s + 2*X2c*Y1s - 2*X1s*Y20 - 2*X1s*Y2c + 2*X1c*Y2s + X1c*X1s*Y1c*k1 - X1c*X1c*Y1s*k1 + X1s*Y1c*Y1c*k2 - X1c*Y1c*Y1s*k2)

    g20 = -4*lp*X2s*Y2c + 4*lp*X2c*Y2s + 2*lp*X1s*X20*Y1c*k1 \
          -lp*X1s*X2c*Y1c*k1 + lp*X1c*X2s*Y1c*k1 - 2*lp*X1c*X20*Y1s*k1 \
          -lp*X1c*X2c*Y1s*k1 -lp*X1s*X2s*Y1s*k1 + 2*lp*X1c*X1s*Y2c*k1 \
          -lp*X1c*X1c*Y2s*k1 + lp*X1s*X1s*Y2s*k1 + lp*X2s*Y1c*Y1c*k2 \
          -2*lp*X2c*Y1c*Y1s*k2 - lp*X2s*Y1s*Y1s*k2 + 2*lp*X1s*Y1c*Y20*k2 \
          -2*lp*X1c*Y1s*Y2c*k2 - lp*X1s*Y1c*Y2c*k2 + lp*X1c*Y1s*Y2c*k2 \
          -lp*X1c*Y1c*Y2s*k2 + lp*X1s*Y1s*Y2s*k2 + 2*lp*X1c*X1s*Z2c*k3 \
          +2*lp*Y1c*Y1s*Z2c*k3 - lp*X1c*X1c*Z2s*k3 + lp*X1s*X1s*Z2s*k3 \
          -lp*Y1c*Y1c*Z2s*k3 + lp*Y1s*Y1s*Z2s*k3 - Y1s*Z20*d_X1c_d_varphi \
          -Y1s*Z2c*d_X1c_d_varphi + Y1c*Z2s*d_X1c_d_varphi + Y1c*Z20*d_X1s_d_varphi \
          -Y1c*Z2c*d_X1s_d_varphi - Y1s*Z2s*d_X1s_d_varphi + X1s*Z20*d_Y1c_d_varphi \
          +X1s*Z2c*d_Y1s_d_varphi - X1c*Z2s*d_Y1c_d_varphi - X1s*Y1c*d_Z20_d_varphi \
          +X1c*Z2c*d_Y1s_d_varphi + X1s*Z2s*d_Y1s_d_varphi - X1s*Y1c*d_Z20_d_varphi \
          +X1c*Y1s*d_Z20_d_varphi
    
    g2c = -4 * lp * X2s * Y20 \
        + 4 * lp * X20 * Y2s \
    - lp * X1s * X20 * Y1c * k1 \
    + 2 * lp * X1s * X2c * Y1c * k1 \
    + lp * X1c * X2s * Y1c * k1 \
    - lp * X1c * X20 * Y1s * k1 \
    - 2 * lp * X1c * X2c * Y1s * k1 \
    + lp * X1s * X2s * Y1s * k1 \
    + 2 * lp * X1c * X1s * Y20 * k1 \
    - lp * X1c*X1c * Y2s * k1 \
    - lp * X1s*X1s * Y2s * k1 \
    + lp * X2s * Y1c*Y1c * k2 \
    - 2 * lp * X20 * Y1c * Y1s * k2 \
    + lp * X2s * Y1s*Y1s * k2 \
    + lp * X1s * Y1c * Y20 * k2 \
    + lp * X1c * Y1s * Y20 * k2 \
    + 2 * lp * X1s * Y1c * Y2c * k2 \
    - 2 * lp * X1c * Y1s * Y2c * k2 \
    - lp * X1c * Y1c * Y2s * k2 \
    - lp * X1s * Y1s * Y2s * k2 \
    + 2 * lp * X1c * X1s * Z20 * k3 \
    + 2 * lp * Y1c * Y1s * Z20 * k3 \
    - lp * X1c*X1c * Z2s * k3 \
    - lp * X1s*X1s * Z2s * k3 \
    - lp * Y1c*Y1c * Z2s * k3 \
    - lp * Y1s*Y1s * Z2s * k3 \
    - Y1s * Z20 * d_X1c_d_varphi \
    - Y1s * Z2c * d_X1c_d_varphi \
    + Y1c * Z2s * d_X1c_d_varphi \
    - Y1c * Z20 * d_X1s_d_varphi \
    + Y1c * Z2c * d_X1s_d_varphi \
    + Y1s * Z2s * d_X1s_d_varphi \
    + X1s * Z20 * d_Y1c_d_varphi \
    + X1s * Z2c * d_Y1c_d_varphi \
    - X1c * Z2s * d_Y1c_d_varphi \
    + X1c * Z20 * d_Y1s_d_varphi \
    - X1c * Z2c * d_Y1s_d_varphi \
    - X1s * Z2s * d_Y1s_d_varphi \
    - X1s * Y1c * d_Z2c_d_varphi \
    + X1c * Y1s * d_Z2c_d_varphi
    
    g2s = 4 * lp * X2c * Y20 \
    - 4 * lp * X20 * Y2c \
    + lp * X1c * X20 * Y1c * k1 \
    - lp * X1c * X2c * Y1c * k1 \
    + 2 * lp * X1s * X2s * Y1c * k1 \
    - lp * X1s * X20 * Y1s * k1 \
    - lp * X1s * X2c * Y1s * k1 \
    - 2 * lp * X1c * X2s * Y1s * k1 \
    - lp * X1c**2 * Y20 * k1 \
    + lp * X1s**2 * Y20 * k1 \
    + lp * X1c**2 * Y2c * k1 \
    + lp * X1s**2 * Y2c * k1 \
    + lp * X20 * Y1c**2 * k2 
    - lp * X2c * Y1c**2 * k2 \
    - lp * X20 * Y1s**2 * k2 \
    - lp * X2c * Y1s**2 * k2 \
    - lp * X1c * Y1c * Y20 * k2 \
    + lp * X1s * Y1s * Y20 * k2 \
    + lp * X1c * Y1c * Y2c * k2 \
    + lp * X1s * Y1s * Y2c * k2 \
    + 2 * lp * X1s * Y1c * Y2s * k2 \
    - 2 * lp * X1c * Y1s * Y2s * k2 \
    - lp * X1c**2 * Z20 * k3 \
    + lp * X1s**2 * Z20 * k3 \
    - lp * Y1c**2 * Z20 * k3 \
    + lp * Y1s**2 * Z20 * k3 \
    + lp * X1c**2 * Z2c * k3 \
    + lp * X1s**2 * Z2c * k3 \
    + lp * Y1c**2 * Z2c * k3 \
    + lp * Y1s**2 * Z2c * k3 \
    + Y1c * Z20 * d_X1c_d_varphi \
    - Y1c * Z2c * d_X1c_d_varphi \
    - Y1s * Z2s * d_X1c_d_varphi \
    - Y1s * Z20 * d_X1s_d_varphi \
    - Y1s * Z2c * d_X1s_d_varphi \
    + Y1c * Z2s * d_X1s_d_varphi \
    - X1c * Z20 * d_Y1c_d_varphi \
    + X1c * Z2c * d_Y1c_d_varphi \
    + X1s * Z2s * d_Y1c_d_varphi \
    + X1s * Z20 * d_Y1s_d_varphi \
    + X1s * Z2c * d_Y1s_d_varphi \
    - X1c * Z2s * d_Y1s_d_varphi \
    - X1s * Y1c * d_Z2s_d_varphi \
    + X1c * Y1s * d_Z2s_d_varphi

    # We consider the system sqrt(g) = 0 and
    # d (sqrtg) / d theta = 0.
    # We algebraically eliminate r in "20200322-02 Max r for Garren Boozer.nb", in the section
    # "Keeping first 3 orders in the Jacobian".
    # We end up with the form in "20200322-01 Max r for GarrenBoozer.docx":
    # K0 + K2s*sin(2*theta) + K2c*cos(2*theta) + K4s*sin(4*theta) + K4c*cos(4*theta) = 0.

    K0 = 2*g1c*g1c*g20 - 3*g1c*g1c*g2c + 8*g0*g2c*g2c + 8*g0*g2s*g2s

    K2s = 2*g1c*g1c*g2s

    K2c = -2*g1c*g1c*g20 + 2*g1c*g1c*g2c

    K4s = g1c*g1c*g2s - 16*g0*g2c*g2s

    K4c = g1c*g1c*g2c - 8*g0*g2c*g2c + 8*g0*g2s*g2s

    coefficients = np.zeros((nphi,5))
    
    coefficients[:, 4] = 4*(K4c*K4c + K4s*K4s)

    coefficients[:, 3] = 4*(K4s*K2c - K2s*K4c)

    coefficients[:, 2] = K2s*K2s + K2c*K2c - 4*K0*K4c - 4*K4c*K4c - 4*K4s*K4s

    coefficients[:, 1] = 2*K0*K2s + 2*K4c*K2s - 4*K4s*K2c

    coefficients[:, 0] = (K0 + K4c)*(K0 + K4c) - K2c*K2c

    for jphi in range(nphi):
        # Solve for the roots of the quartic polynomial:
        try:
            roots = np.polynomial.polynomial.polyroots(coefficients[jphi, :]) # Do I need to reverse the order of the coefficients?
        except np.linalg.LinAlgError:
            raise RuntimeError('Problem with polyroots. coefficients={} lp={} B0={} g0={} g1c={}'.format(coefficients[jphi, :], lp, s.B0, g0, g1c))

        real_parts = np.real(roots)
        imag_parts = np.imag(roots)

        logger.debug('jphi={} g0={} g1c={} g20={} g2s={} g2c={} K0={} K2s={} K2c={} K4s={} K4c={} coefficients={} real={} imag={}'.format(jphi, g0[jphi], g1c[jphi], g20[jphi], g2s[jphi], g2c[jphi], K0[jphi], K2s[jphi], K2c[jphi], K4s[jphi], K4c[jphi], coefficients[jphi,:], real_parts, imag_parts))

        # This huge number indicates a true solution has not yet been found.
        rc = 1e+100

        for jr in range(4):
            # Loop over the roots of the equation for w.

            # If root is not purely real, skip it.
            if np.abs(imag_parts[jr]) > 1e-7:
                logger.debug("Skipping root with jr={} since imag part is {}".format(jr, imag_parts[jr]))
                continue

            sin2theta = real_parts[jr]

            # Discard any roots that have magnitude larger than 1. (I'm not sure this ever happens, but check to be sure.)
            if np.abs(sin2theta) > 1:
                logger.debug("Skipping root with jr={} since sin2theta={}".format(jr, sin2theta))
                continue

            # Determine varpi by checking which choice gives the smaller residual in the K equation
            abs_cos2theta = np.sqrt(1 - sin2theta * sin2theta)
            residual_if_varpi_plus  = np.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta \
                                             + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
            residual_if_varpi_minus = np.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) \
                                             + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))

            if residual_if_varpi_plus > residual_if_varpi_minus:
                varpi = -1
            else:
                varpi = 1

            cos2theta = varpi * abs_cos2theta

            # The next few lines give an older method for computing varpi, which has problems in edge cases
            # where w (the root of the quartic polynomial) is very close to +1 or -1, giving varpi
            # not very close to +1 or -1 due to bad loss of precision.
            #
            #varpi_denominator = ((K4s*2*sin2theta + K2c) * sqrt(1 - sin2theta*sin2theta))
            #if (abs(varpi_denominator) < 1e-8) print *,"WARNING!!! varpi_denominator=",varpi_denominator
            #varpi = -(K0 + K2s * sin2theta + K4c*(1 - 2*sin2theta*sin2theta)) / varpi_denominator
            #if (abs(varpi*varpi-1) > 1e-3) print *,"WARNING!!! abs(varpi*varpi-1) =",abs(varpi*varpi-1)
            #varpi = nint(varpi) ! Ensure varpi is exactly either +1 or -1.
            #cos2theta = varpi * sqrt(1 - sin2theta*sin2theta)

            # To get (sin theta, cos theta) from (sin 2 theta, cos 2 theta), we consider two cases to
            # avoid precision loss when cos2theta is added to or subtracted from 1:
            get_cos_from_cos2 = cos2theta > 0
            if get_cos_from_cos2:
                abs_costheta = np.sqrt(0.5*(1 + cos2theta))
            else:
                abs_sintheta = np.sqrt(0.5 * (1 - cos2theta))
                
            logger.debug("  jr={}  sin2theta={}  cos2theta={}".format(jr, sin2theta, cos2theta))
            for varsigma in [-1, 1]:
                if get_cos_from_cos2:
                    costheta = varsigma * abs_costheta
                    sintheta = sin2theta / (2 * costheta)
                else:
                    sintheta = varsigma * abs_sintheta
                    costheta = sin2theta / (2 * sintheta)
                logger.debug("    varsigma={}  costheta={}  sintheta={}".format(varsigma, costheta, sintheta))

                # Sanity test
                if np.abs(costheta*costheta + sintheta*sintheta - 1) > 1e-13:
                    msg = "Error! sintheta={} costheta={} jphi={} jr={} sin2theta={} cos2theta={} abs(costheta*costheta + sintheta*sintheta - 1)={}".format(sintheta, costheta, jphi, jr, sin2theta, cos2theta, np.abs(costheta*costheta + sintheta*sintheta - 1))
                    logger.error(msg)
                    raise RuntimeError(msg)

                """
                # Try to get r using the simpler method, the equation that is linear in r.
                denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                if np.abs(denominator) > 1e-8:
                    # This method cannot be used if we would need to divide by 0
                    rr = g1c[jphi] * sintheta / denominator
                    residual = g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta) # Residual in the equation sqrt(g)=0.
                    logger.debug("    Linear method: rr={}  residual={}".format(rr, residual))
                    if (rr>0) and np.abs(residual) < 1e-5:
                        if rr < rc:
                            # If this is a new minimum
                            rc = rr
                            sintheta_at_rc = sintheta
                            costheta_at_rc = costheta
                            logger.debug("      New minimum: rc={}".format(rc))
                else:
                    # Use the more complicated method to determine rr by solving a quadratic equation.
                    quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
                    quadratic_B = costheta * g1c[jphi]
                    quadratic_C = g0[jphi]
                    radical = np.sqrt(quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C)
                    for sign_quadratic in [-1, 1]:
                        rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
                        residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                        logger.debug("    Quadratic method: A={} B={} C={} radicand={}, radical={}  rr={}  residual={}".format(quadratic_A, quadratic_B, quadratic_C, quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C, radical, rr, residual))
                        if (rr>0) and np.abs(residual) < 1e-5:
                            if rr < rc:
                                # If this is a new minimum
                                rc = rr
                                sintheta_at_rc = sintheta
                                costheta_at_rc = costheta
                                logger.debug("      New minimum: rc={}".format(rc))
                """

                # Try to get r using the simpler method, the equation that is linear in r.
                linear_solutions = []
                denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                if np.abs(denominator) > 1e-8:
                    # This method cannot be used if we would need to divide by 0
                    rr = g1c[jphi] * sintheta / denominator
                    residual = g0[jphi] + rr * g1c[jphi] * costheta + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta) # Residual in the equation sqrt(g)=0.
                    logger.debug("    Linear method: rr={}  residual={}".format(rr, residual))
                    if (rr>0) and np.abs(residual) < 1e-5:
                        linear_solutions = [rr]
                        
                # Use the more complicated method to determine rr by solving a quadratic equation.
                quadratic_solutions = []
                quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
                quadratic_B = costheta * g1c[jphi]
                quadratic_C = g0[jphi]
                radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
                if np.abs(quadratic_A) < 1e-13:
                    rr = -quadratic_C / quadratic_B
                    residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                    logger.debug("    Quadratic method but A is small: A={} rr={}  residual={}".format(quadratic_A, rr, residual))
                    if rr > 0 and np.abs(residual) < 1e-5:
                        quadratic_solutions.append(rr)
                else:
                    # quadratic_A is nonzero, so we can apply the quadratic formula.
                    # I've seen a case where radicand is <0 due I think to under-resolution in phi.
                    if radicand >= 0:
                        radical = np.sqrt(radicand)
                        for sign_quadratic in [-1, 1]:
                            rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A) # This is the quadratic formula.
                            residual = -g1c[jphi] * sintheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta) # Residual in the equation d sqrt(g) / d theta = 0.
                            logger.debug("    Quadratic method: A={} B={} C={} radicand={}, radical={}  rr={}  residual={}".format(quadratic_A, quadratic_B, quadratic_C, quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C, radical, rr, residual))
                            if (rr>0) and np.abs(residual) < 1e-5:
                                quadratic_solutions.append(rr)

                logger.debug("    # linear solutions={}  # quadratic solutions={}".format(len(linear_solutions), len(quadratic_solutions)))
                if len(quadratic_solutions) > 1:
                    # Pick the smaller one
                    quadratic_solutions = [np.min(quadratic_solutions)]
                    
                # If both methods find a solution, check that they agree:
                if len(linear_solutions) > 0 and len(quadratic_solutions) > 0:
                    diff = np.abs(linear_solutions[0] - quadratic_solutions[0])
                    logger.debug("  linear solution={}  quadratic solution={}  diff={}".format(linear_solutions[0], quadratic_solutions[0], diff))
                    if diff > 1e-5:
                        warnings.warn("  Difference between linear solution {} and quadratic solution {} is {}".format(linear_solutions[0], quadratic_solutions[0], diff))
                        
                    #assert np.abs(linear_solutions[0] - quadratic_solutions[0]) < 1e-5, "Difference between linear solution {} and quadratic solution {} is {}".format(linear_solutions[0], quadratic_solutions[0], linear_solutions[0] - quadratic_solutions[0])
                    
                # Prefer the quadratic solution
                rr = -1
                if len(quadratic_solutions) > 0:
                    rr = quadratic_solutions[0]
                elif len(linear_solutions) > 0:
                    rr = linear_solutions[0]

                if rr > 0 and rr < rc:
                    # This is a new minimum
                    rc = rr
                    sintheta_at_rc = sintheta
                    costheta_at_rc = costheta
                    logger.debug("      New minimum: rc={}".format(rc))
                    
        r_singularity_basic_vs_varphi[jphi] = rc
        #r_singularity_Newton_solve()
        r_singularity_vs_varphi[jphi] = rc
        r_singularity_residual_sqnorm[jphi] = 0 # Newton_residual_sqnorm
        r_singularity_theta_vs_varphi[jphi] = 0 # theta FIX ME!!

    self.r_singularity_vs_varphi = r_singularity_vs_varphi
    self.inv_r_singularity_vs_varphi = 1 / r_singularity_vs_varphi
    self.r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi
    self.r_singularity = np.min(r_singularity_vs_varphi)    
    self.r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi
    self.r_singularity_residual_sqnorm = r_singularity_residual_sqnorm