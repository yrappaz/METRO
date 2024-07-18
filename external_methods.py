from scipy.integrate import simps
from packages import *
from physical_units import *

from power_spectrum_methods import *
from scipy.optimize import curve_fit
import colorsys

def calculate_rms(array):

    if isinstance(array, np.ndarray) == False:
        raise Exception("The argument should be a Numpy array of dimension(s) 1, 2 or 3.")
    else:
        ndim = len(array.shape)
        if ndim == 1:

            L = array.shape[0]
            temp = array**2
            return np.sqrt((1/L)*np.sum(temp))

        elif ndim == 2:

            L1 = array.shape[0]
            L2 = array.shape[1]
            temp = array ** 2
            return np.sqrt((1 /(L1*L2)) * np.sum(temp))

        elif ndim == 3:

            L1 = array.shape[0]
            L2 = array.shape[1]
            L3 = array.shape[2]
            temp = array ** 2
            return np.sqrt((1 / (L1*L2*L3)) * np.sum(temp))

def extract_bfield_data(data_path):

    raw_data_B = np.loadtxt(data_path)

    ngrid = np.max(raw_data_B[:,0].astype(int))+1

    bx = np.empty((ngrid, ngrid, ngrid), dtype=float)
    by = np.empty((ngrid, ngrid, ngrid), dtype=float)
    bz = np.empty((ngrid, ngrid, ngrid), dtype=float)
    #
    for i in range(0, len(raw_data_B)):

        cx, cy, cz = int(raw_data_B[i][0]), int(raw_data_B[i][1]), int(raw_data_B[i][2])

        bx[cx, cy, cz] = raw_data_B[i][3]
        by[cx, cy, cz] = raw_data_B[i][4]
        bz[cx, cy, cz] = raw_data_B[i][5]

    return bx,by,bz


















def c_for_early_halos(M):

    # return 15
    return 4.67*(M/(1e14*solar_masses_to_gr))**(-0.11)

def Lambda(x,rs):

    return np.log(1+x/rs)-(x/(x+rs))

def Theta(x,rs):
    temp = (x/(rs*(rs+x)))+((np.log((rs+x)/rs)**2)/(x))
    return temp

def Delta(x,rs):

    return (rs/2)-((rs**2*(2*(x+rs)*np.log((rs+x)/rs))+rs)/(2*(rs+x)**2))

def rho_s(M,rv,rs):

    return M/(4*math.pi*(rs**3)*Lambda(rv,rs))

def Sigma(r,rs):

    temp_1 = np.log(1+r/rs)-(r/(r+rs))
    temp_2 = 1/((r**3)*(1+r/rs)**2)

    return temp_1*temp_2

def B_const(rhos, rs, T0):

    return (4*math.pi*G*rhos*(rs**2)*0.5*mp)/(kb*T0)

def compute_potential_energy(M, rv, rs):

    return -(G/2)*(M**2)*((Theta(rv,rs)/Lambda(rv,rs))+(1/rv))

def compute_kinetic_energy(M, rv, rs):

    contribution_1 = (rv**2)*G*M*(rho_s(M,rv,rs)/((rv/rs)*(1+(rv/rs))**2))

    contribution_2 = 4*math.pi*G*(rho_s(M,rv,rs)**2)*(rs**4)*Delta(rv,rs)

    return 2*math.pi*(contribution_1+contribution_2)

def compute_total_individual_energy(M, rv, rs):

    return compute_potential_energy(M, rv, rs)+compute_kinetic_energy(M, rv, rs)

def compute_energy_merging_haloes(array_of_mass, array_of_rv, array_of_rs, Mf, rvf):

    L = len(array_of_mass)

    # ************ Computation of individual potential + kinetic energies ************
    INDIVIDUAL_ENERGIES = 0

    for i in range(0, L):

        temp_e_pot = compute_potential_energy(array_of_mass[i], array_of_rv[i], array_of_rs[i])
        temp_e_kin = compute_kinetic_energy(array_of_mass[i], array_of_rv[i], array_of_rs[i])

        INDIVIDUAL_ENERGIES += (temp_e_pot+temp_e_kin)
    # ********************************************************************************


    # ************** Compute gravitational potential energy between halos ************
    GRAV_POT_ENERGY = 0

    halos = np.arange(0, L)
    comb = list(it.combinations(halos, 2))

    len_comb = len(comb)

    for i in range(0, len_comb):

        ind1 = comb[i][0]
        ind2 = comb[i][1]

        temp = -G*array_of_mass[ind1]*array_of_mass[ind2]/(5*(array_of_rv[ind1]+array_of_rv[ind2]))

        GRAV_POT_ENERGY += temp
    # ********************************************************************************


    # ********** Compute the energy associated with accreted matter ******************
    sum_M = np.sum(array_of_mass)
    delta_M = Mf-sum_M

    ACCR_ENERGY = -G*sum_M*delta_M/rvf
    # ********************************************************************************

    return INDIVIDUAL_ENERGIES+GRAV_POT_ENERGY+ACCR_ENERGY

def compute_c_and_rs_from_energy_conservation(array_of_mass, array_of_rv, array_of_rs, Mf, rvf):

    multiple_c = np.logspace(-3,3,10000)
    E_progenitors = compute_energy_merging_haloes(array_of_mass, array_of_rv, array_of_rs, Mf, rvf)
    E_final = compute_total_individual_energy(Mf, rvf, rvf/multiple_c)

    ratio = E_final/E_progenitors

    indices = np.argwhere((ratio > 0.9) & (ratio < 1.1))

    if len(indices) == 0:
        print('PROBLEM')
    temp_c = np.take(multiple_c, indices)

    c_avg = np.mean(temp_c)

    rs_avg = rvf/c_avg

    return c_avg, rs_avg







def determine_merger_type(mass, array_mass_parents, threshold = 0.1):

    L = len(array_mass_parents)
    max_mass = np.max(array_mass_parents)
    if max_mass/mass >= threshold:
        return 'major'
    else:
        return 'minor'

def adjust_turbulent_velocity_and_injection_scale_decay(zi, zf, init_r, init_vturb, init_L0, init_density_gas,  r_range_new, density_gas_new, temperature_new, particle_density_new):

    cosmo = LambdaCDM(H0=cosmology_params['H0'], Om0=cosmology_params['OM_m'], Ode0=cosmology_params['OM_lambda'])

    ti = cosmo.age(zi).value*gyr_to_s
    tf = cosmo.age(zf).value*gyr_to_s

    rvir = init_r[len(init_r)-1]
    initial_turbulent_energy = simps(2*math.pi*(init_r ** 2)*init_density_gas*init_vturb**2, init_r)

    # Decay of kinetic energy
    new_turbulent_energy = ((tf/ti)**(-6/5))*initial_turbulent_energy

    # Calculating the corresponding new v0 values
    rvir_new = r_range_new[len(r_range_new) - 1]
    integral_new = simps(2*math.pi*(r_range_new ** 2) * density_gas_new * (1 + r_range_new / rvir_new), r_range_new)

    v0 = np.sqrt(new_turbulent_energy / integral_new)
    new_velocity = v0 * np.sqrt(1 + r_range_new / rvir_new)

    # Decay of forcing scale
    new_L0 = ((tf/ti) ** (2/5)) * init_L0

    return new_L0, new_velocity

def get_radial_average_from_matrix(matrix, rvir, abs_val):

    if abs_val == True:
        matrix = np.abs(matrix)

    grid_size = (2*rvir)/matrix.shape[0]

    center_x = int(matrix.shape[0]/2)
    center_y = int(matrix.shape[1]/2)

    dist_max = matrix.shape[0] - center_x

    y_coords, x_coords = np.indices(matrix.shape)
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    bins = np.arange(0, dist_max+2, 1)

    radial_profile, _ = np.histogram(distances, bins=bins, weights=matrix)
    counts, _ = np.histogram(distances, bins=bins)
    counts[counts == 0] = 1

    radial_average = radial_profile / counts
    r_range = np.arange(len(radial_average))*grid_size

    return r_range, radial_average

def fit_power_law_rm_power_spectrum(rm):

    ps = get_spectrum(rm, ncmp=1)

    ind_max = np.argmax(ps['P_tot'])
    kmax = ps['k'][ind_max]

    # print(kmax)

    y_to_fit_log = np.log10(ps['P_tot'][1:ind_max])
    x_to_fit_log = np.log10(ps['k'][1:ind_max])

    def lin_fit(x, m, b):
        return m*x+b

    # Ajustement de la fonction aux données
    params, covariance = curve_fit(lin_fit, x_to_fit_log, y_to_fit_log)

    # Paramètres ajustés
    m_fit, b_fit = params

    return kmax, m_fit

def fit_power_law_random_data(xdata, ydata):

    def power_law_fit(x, a,p,c):
        return a*(x**p)+c

    # Ajustement de la fonction aux données
    params, covariance = curve_fit(power_law_fit, xdata, ydata)

    # Paramètres ajustés
    a_fit, p_fit, c_fit = params

    return a_fit, p_fit, c_fit

def extract_submatrices(matrix, submatrix_size):
    num_rows, num_cols = matrix.shape
    submatrix_shape = (num_rows - submatrix_size + 1, num_cols - submatrix_size + 1, submatrix_size, submatrix_size)
    submatrix_strides = matrix.strides * 2

    submatrices = np.lib.stride_tricks.as_strided(matrix, shape=submatrix_shape, strides=submatrix_strides)

    return submatrices.reshape((-1, submatrix_size, submatrix_size))

def calculate_std_array_of_matrices(data, shape):

    eff_data = np.reshape(data, (shape[0], shape[1]*shape[2]))
    std = np.std(eff_data, axis = 1)
    return std

def linear_fit(xdata, ydata):

    def lin_fit(x, a,b):
        return a*x+b

    # Ajustement de la fonction aux données
    params, covariance = curve_fit(lin_fit, xdata, ydata)

    # Paramètres ajustés
    # a_fit, b_fit = params

    return params, covariance

def generate_random_color():
    while True:
        # Generate a random hue in the range [0, 1)
        hue = random.random()

        # Set a minimum and maximum value for saturation and luminance
        min_saturation = 0.4
        max_saturation = 0.8
        min_luminance = 0.4
        max_luminance = 0.6

        # Generate random saturation and luminance values within the specified range
        saturation = random.uniform(min_saturation, max_saturation)
        luminance = random.uniform(min_luminance, max_luminance)

        # Convert HSL to RGB
        r, g, b = [int(256 * c) for c in colorsys.hls_to_rgb(hue, luminance, saturation)]

        return r / 255, g / 255, b / 255  # Return RGB values between 0 and 1









