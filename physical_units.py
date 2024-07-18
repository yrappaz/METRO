# from astropy import constants as const
# from astropy import units as u
# from astropy.cosmology import WMAP9 as cosmo
# import math
from packages import *

# # ///////////// PHYSICAL CONSTANTS ///////////////
me = (const.m_e.cgs).value
mp = (const.m_p.cgs).value
e = (const.e.gauss).value
sigma_t = (const.sigma_T.cgs).value
c = (const.c.cgs).value
kb = (const.k_B.cgs).value
h = (const.h.cgs).value
PI = math.pi
mc = 12*mp
mo = 16*mp
G = (const.G.cgs).value
R = (const.R.cgs).value
# # ////////////////////////////////////////////////

# # ///////////// CONVERSION FACTORS ///////////////
kpc_to_cm = 3.08e21
cm_to_kpc = 1/kpc_to_cm
pc_to_cm = 3.086e18
cm_to_pc = 1/pc_to_cm

myr_to_s = 1e6*365*24*3600
s_to_myr = 1/myr_to_s

km_to_mpc = 3.24078e-20
mpc_to_cm = 3.086e24
cm_to_mpc = 1/mpc_to_cm

gyr_to_s = 1e9*365*24*3600
s_to_gyr = 1/gyr_to_s

solar_masses_to_gr = 1.989e33
gr_to_solar_masses = 1/(solar_masses_to_gr)

K_to_eV = 1/11606
K_to_kev = 8.6173e-8
# //////////////////////////////////////////////////