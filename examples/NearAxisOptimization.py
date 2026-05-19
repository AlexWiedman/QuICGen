from QuICGen import QIC
from scipy.optimize import minimize, least_squares
import numpy as np
from Optimizable import B1c_hat_even_calc, B1s_hat_even_calc, elongation_constraints, iota_constraints, minimum_major_radius, B_variance_appx
from StellConfig import StellConfig
from QuICGen.plot import plot, plot_boundary, B_contour, B_fieldline
import time
import matplotlib.pyplot as plt
import os
import csv

# Random number generater seed
seed = 66#np.random.randint(0, 0xfff)
rng = np.random.default_rng(seed)

# Number of iterations (only applicable for BFGS method)
MAXITER = 5000

# Initial values, does not change during optimization
nfourier = 7
nphi = 61
nfp = 2
r_max = 0.05

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# Initial values for Axis Shape
rc = np.zeros(nfourier)
zs = np.zeros(nfourier)
rc[0] = 1
zs[0] = 0

# Initial values for X1c and Y1c fourier series
x1c_fourier = np.zeros(nfourier)
y1c_fourier = np.zeros(nfourier)
x1c_fourier[0] = 0.5
y1c_fourier[0] = 0.5

# Initial value of Epsilon for B0 calculation
epsilon = 0.2

# Call the StellConfig class to store the DOFS and perform necessary calculations
dofs_class = StellConfig(nfourier, nphi, nfp, rc=rc, zs=zs, x1c_fourier=x1c_fourier, y1c_fourier=y1c_fourier, epsilon=epsilon)

#
scale = dofs_class.bounds[0]

# Set rc, zs, x1c, and y1c values randomly (based off of a seed)
rc[1:] = (2 * rng.random(nfourier-1) - 1) * scale[0:nfourier-1]
scale = scale[nfourier-1:]

zs[1:] = (2 * rng.random(nfourier-1) - 1) * scale[0:nfourier-1]
scale = scale[nfourier-1:]

x1c_fourier = (2 * rng.random(nfourier) - 1) * scale[0:nfourier]
scale = scale[nfourier:]

y1c_fourier = (2 * rng.random(nfourier) - 1) * scale[0:nfourier]
scale = scale[nfourier:]

# Reset StellConfig class to new values
dofs_class = StellConfig(nfourier, nphi, nfp, rc=rc, zs=zs, x1c_fourier=x1c_fourier, y1c_fourier=y1c_fourier, epsilon=epsilon)

# Call the QIC class to 
stel = QIC(dofs_class.rc, dofs_class.zs, nfp = nfp, X1c=dofs_class.X1c, Y1c=dofs_class.Y1c, B0=dofs_class.B0, nphi = nphi)

# Save the initial parameters for the stellarator
save_array = dofs_class.construct_save_array()
timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
np.save(OUT_DIR + f'Initial_stel_{timestamp}', save_array)

# Define optimization function
def fun_ls(dofs):

    dofs_class.set_dofs(dofs)
    stel = QIC(dofs_class.rc, dofs_class.zs, nfp = dofs_class.nfp, X1c=dofs_class.X1c, Y1c=dofs_class.Y1c, B0=dofs_class.B0, nphi = dofs_class.nphi)

    avg_elongation, elongation_square  = elongation_constraints(stel.X1c, stel.Y1c, stel.X1s, stel.Y1s, constraint = 4)
    iota_err = iota_constraints(stel.iota)
    B1c_hat_even = B1c_hat_even_calc(stel.B1s, stel.B1c, stel.iota, stel.nphi, stel.nfp)
    B1s_hat_even = B1s_hat_even_calc(stel.B1s, stel.B1c, stel.iota, stel.nphi, stel.nfp)
    B_var_max = B_variance_appx(stel.B1s[0], stel.B1c[0], r_max)
    B_var_min = B_variance_appx(stel.B1s[int(nphi/2)], stel.B1c[int(nphi/2)], r_max)

    J = B1c_hat_even + B1s_hat_even + iota_err + 1e1*elongation_square #+ 1e5*B_var_max + 1e5*B_var_min
    print(f"LS J:  {np.sum(np.pow(J, 2)):.2e},    B1c_hat_even:  {np.sum(np.pow(B1c_hat_even, 2)):.2e},    B1s_hat_even:   {np.sum(np.pow(B1s_hat_even, 2)):.2e}   iota:  {stel.iota:.2e},  Average Elongation: {avg_elongation:.2e}")
    
    return J

def fun_bfgs(dofs):

    dofs_class.set_dofs(dofs)
    stel = QIC(dofs_class.rc, dofs_class.zs, nfp = dofs_class.nfp, X1c=dofs_class.X1c, Y1c=dofs_class.Y1c, B0=dofs_class.B0, nphi = dofs_class.nphi)

    avg_elongation, elongation_square  = elongation_constraints(stel.X1c, stel.Y1c, stel.X1s, stel.Y1s, constraint = 4)
    iota_err = iota_constraints(stel.iota)
    B1c_hat_even = B1c_hat_even_calc(stel.B1s, stel.B1c, stel.iota, stel.nphi, stel.nfp)
    B1s_hat_even = B1s_hat_even_calc(stel.B1s, stel.B1c, stel.iota, stel.nphi, stel.nfp)
    B_var_max = B_variance_appx(stel.B1s[0], stel.B1c[0], r_max)
    B_var_min = B_variance_appx(stel.B1s[int(nphi/2)], stel.B1c[int(nphi/2)], r_max)

    J = np.sum(np.pow(B1c_hat_even,2)) + np.sum(np.pow(B1s_hat_even, 2)) + 1e2*iota_err + np.sum(np.pow(elongation_square,2)) + B_var_max + B_var_min
    print(f"J:  {J:.2e},    B1c_hat_even:  {np.sum(np.pow(B1c_hat_even, 2)):.2e},    B1s_hat_even:   {np.sum(np.pow(B1s_hat_even, 2)):.2e}   iota:  {stel.iota:.2e},  Average Elongation: {avg_elongation:.2e}")
    
    return J

dofs = dofs_class.get_dofs()

# Perform a Least Squares optimization
res = least_squares(fun_ls, dofs, jac='2-point', bounds=dofs_class.bounds)

# Perform a BFGS optimization
#res = minimize(fun_bfgs, dofs, jac=False, method='BFGS', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)


dofs_class.set_dofs(res.x)
QI_stel = QIC(dofs_class.rc, dofs_class.zs, nfp = dofs_class.nfp, X1c=dofs_class.X1c, Y1c=dofs_class.Y1c, B0=dofs_class.B0, nphi = dofs_class.nphi)

save_array = dofs_class.construct_save_array()
np.save(OUT_DIR + f'QI_stel_{timestamp}_{seed}', save_array)

# Plotting

plot(QI_stel)
plot_boundary(QI_stel, nphi=nphi, r=r_max)
B_contour(QI_stel, r=r_max, nphi=nphi, show=False)

alpha = np.linspace(-2, 4, 16)
x_plt = np.linspace(0, 2, nphi)
for i in alpha:
    plt.plot(x_plt, QI_stel.iota * x_plt + i, label = f"Alpha: {i:.2}", color = 'r')
plt.ylim(0, 2)
plt.show()
