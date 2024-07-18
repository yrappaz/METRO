# //////////////////////////////////////////////////////////////////////////////////////////////
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.io import fits
import math
import time
from scipy import integrate
from scipy.integrate import romberg
from scipy.special import gamma

import csv
from scipy.ndimage.interpolation import rotate
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams["font.family"] = "Cambria Math"
import scipy
from scipy.integrate import quad
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterExponent



from astropy import constants as const
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from physical_units import *





# H0 = 70
# OM_r = 0
# OM_m = 0.3
# OM_k = 0
# OM_lambda = 0.7
#
#
#

H0 = 67.4
OM_r = 0
OM_m = 0.315
OM_k = 0
OM_lambda = 0.685


def cosmological_time_from_redshift(z):     # unit: Gyr

    temp = 28/(1+(1+z)**2)

    return temp

def cosmological_time_from_planck(arguments):
    z = float(arguments[0])  # redshift
    H0 = float(arguments[1])  # Hubble constant
    WM = float(arguments[2])  # Omega(matter)
    WV = float(arguments[3])  # Omega(vacuum) or lambda

    WR = 0.  # Omega(radiation)
    WK = 0.  # Omega curvaturve = 1-Omega(total)
    c = 299792.458  # velocity of light in km/sec
    Tyr = 977.8  # coefficent for converting 1/H into Gyr
    DTT = 0.5  # time from z to now in units of 1/H0
    DTT_Gyr = 0.0  # value of DTT in Gyr
    age = 0.5  # age of Universe in units of 1/H0
    age_Gyr = 0.0  # value of age in Gyr
    zage = 0.1  # age of Universe at redshift z in units of 1/H0
    zage_Gyr = 0.0  # value of zage in Gyr
    DCMR = 0.0  # comoving radial distance in units of c/H0
    DCMR_Mpc = 0.0
    DCMR_Gyr = 0.0
    DA = 0.0  # angular size distance
    DA_Mpc = 0.0
    DA_Gyr = 0.0
    kpc_DA = 0.0
    DL = 0.0  # luminosity distance
    DL_Mpc = 0.0
    DL_Gyr = 0.0  # DL in units of billions of light years
    V_Gpc = 0.0
    a = 1.0  # 1/(1+z), the scale factor of the Universe
    az = 0.5  # 1/(1+z(object))

    h = H0 / 100.
    WR = 4.165E-5 / (h * h)  # includes 3 massless neutrino species, T0 = 2.72528
    WK = 1 - WM - WR - WV
    az = 1.0 / (1 + 1.0 * z)
    age = 0.
    n = 1000  # number of points in integrals
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        age = age + 1. / adot

    zage = az * age / n
    zage_Gyr = (Tyr / H0) * zage
    DTT = 0.0
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DTT = DTT + 1. / adot
        DCMR = DCMR + 1. / (a * adot)

    DTT = (1. - az) * DTT / n
    DCMR = (1. - az) * DCMR / n
    age = DTT + zage
    age_Gyr = age * (Tyr / H0)
    DTT_Gyr = (Tyr / H0) * DTT
    DCMR_Gyr = (Tyr / H0) * DCMR
    DCMR_Mpc = (c / H0) * DCMR

    # tangential comoving distance

    ratio = 1.00
    x = sqrt(abs(WK)) * DCMR
    if x > 0.1:
        if WK > 0:
            ratio = 0.5 * (exp(x) - exp(-x)) / x
        else:
            ratio = sin(x) / x
    else:
        y = x * x
        if WK < 0: y = -y
        ratio = 1. + y / 6. + y * y / 120.
    DCMT = ratio * DCMR
    DA = az * DCMT
    DA_Mpc = (c / H0) * DA
    kpc_DA = DA_Mpc / 206.264806
    DA_Gyr = (Tyr / H0) * DA
    DL = DA / (az * az)
    DL_Mpc = (c / H0) * DL
    DL_Gyr = (Tyr / H0) * DL

    # comoving volume computation

    ratio = 1.00
    x = sqrt(abs(WK)) * DCMR
    if x > 0.1:
        if WK > 0:
            ratio = (0.125 * (exp(2. * x) - exp(-2. * x)) - x / 2.) / (x * x * x / 3.)
        else:
            ratio = (x / 2. - sin(2. * x) / 4.) / (x * x * x / 3.)
    else:
        y = x * x
        if WK < 0: y = -y
        ratio = 1. + y / 5. + (2. / 105.) * y * y
    VCM = ratio * DCMR * DCMR * DCMR / 3.
    V_Gpc = 4. * pi * ((0.001 * c / H0) ** 3) * VCM

    return zage_Gyr

def evolution_scale_factor(z):

    if z == 0:

        return 1

    elif z <= 0.5 and z > 0:

        return 1

    elif z > 0.5:

        temp = (cosmological_time_from_redshift(z)/cosmological_time_from_redshift(0.5))**(2/3)

        return temp

def hubble_parameter_evolution(h0, z, omega_r, omega_m, omega_lambda, omega_k):     # unit: km/s/Mpc

    return np.sqrt((h0**2)*(omega_r*(1+z)**4+omega_m*(1+z)**3+omega_k*(1+z)**2+omega_lambda))

def critical_density_evolution(z, H0, OM_r,OM_m,OM_k,OM_lambda):     # unit: g/cm^3

    return (3*(hubble_parameter_evolution(H0, z, OM_r, OM_m, OM_k, OM_lambda)*km_to_mpc)**2)/(8*PI*G)

def matter_density_evolution(z):

    if z >= 0.5:

        return 1/(6*math.pi*G*(cosmological_time_from_redshift(z)*gyr_to_s)**2)

    elif z < 0.5:

        return 1/(6*math.pi*G*(cosmological_time_from_redshift(0.5)*gyr_to_s)**2)

def calculate_collapsing_time_and_radius(initial_redshift, initial_overdensity, initial_radius):

    Omega_i = matter_density_evolution(initial_redshift)/critical_density_evolution(initial_redshift)

    initial_time = cosmological_time_from_redshift(initial_redshift)*gyr_to_s

    A = 0.5*initial_radius/((5/3)*initial_overdensity+1-(1/Omega_i))
    B = (3/4)*initial_time/((5/3)*initial_overdensity+1-(1/Omega_i))

    tmax = math.pi*B
    rmax = 2*A

    return A, B, tmax, rmax

def solve_radius_and_time(initial_redshift, initial_overdensity, initial_radius, time_to_solve):

    a,b,tmax,rmax = calculate_collapsing_time_and_radius(initial_redshift, initial_overdensity, initial_radius)

    theta = np.linspace(0, math.pi, 1000)

    r = a*(1-np.cos(theta))
    t = b*(theta-np.sin(theta))

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    sol_r_index = find_nearest(t, time_to_solve)

    return r[sol_r_index]

def gaussian_random_density_field(dr_kpc, virial_radius, background_density, mass):

    average_density_subhalo = mass/((4/3)*math.pi*virial_radius**3)
    sigma_square = background_density


    condition = True
    count = 0

    R = np.array([])
    D = np.array([])


    while condition:

        count += 1

        random_comp = np.random.normal(average_density_subhalo, sigma_square, 1)[0]

        temp_r = count*dr_kpc*kpc_to_cm
        temp_dens = random_comp

        R = np.append(R, temp_r)
        D = np.append(D, temp_dens)


        temporary_mass = integrate.simps(4*math.pi*R**2*D, R)

        if temporary_mass > mass:

            condition = False

    return R,D

def virial_radius_evolution(z, virial_mass,  H0, OM_r,OM_m,OM_k,OM_lambda):     # unit: cm

    return ((3*virial_mass)/(4*math.pi*200*critical_density_evolution(z, H0, OM_r,OM_m,OM_k,OM_lambda)))**(1/3)

def calculate_virial_velocity(M, redshift):

    return (163*1e5)*(M/(1e12*solar_masses_to_gr))*((OM_m**(1/6))*(1+redshift)**(1/2))




















#
