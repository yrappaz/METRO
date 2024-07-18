from packages import *

mergertree_params = {
'zmax':4,
'zres':300,
'log_mres':12,
'log_mclust':15,
'nb_real': 1000,
'single_tree_number': 1,
'resolution': 1000,
'gamma': 1.2,
'fb': 0.1,
'alpha_0': 20,
'z_lim_eff': 1.5,
'incr_factor': 1,
'incr_factors': np.arange(1,11)
}

bfield_params = {
'mag_res_fraction': 1e3,
'all_mag_res_fractions': np.array([1e3]),
'all_z_start': np.array([0.5,1,1.5,2.0]),
'z_start': 2.0,
'b0': 1e-20,
'all_b0': np.array([1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20]),
'incr_factor': 1,
'all_incr_factors': np.array([1,2,3,4,5])
}

cosmology_params = {
'H0': 67.4,
'OM_r': 0,
'OM_m': 0.315,
'OM_k': 0,
'OM_lambda': 0.685
}

mult_trees_figures_params = {
'array_zmax': np.array([4,3,2]),
'zres':300,
'array_log_mres': np.array([12,11,10]),
'array_log_mclust':np.array([15,14,13]),
'nb_real': 20,
'z_lim_eff':1.5,
'resolution': 1000,
'single_tree_number': 1
}

statistics_params = {
'log_mclust': np.array([15,14,13]),
'log_mres': np.array([12,11,10]),
'all_zmax': np.array([3,3,3]),
'zres': 300,
'z_lim_eff':1,
'nb_real': 1000
}


all_plasma_quantities =  {
'coll_freq_ii': np.array([r'\nu_{ii}', r'\mathrm{s}^{-1}']),
'density_gas': np.array([r'\rho_{\mathrm{g}}', r'\mathrm{g}~\mathrm{cm}^{-3}']),
'concentration_param': np.array([r'c', '-']),
'density_dm': np.array([r'\rho_{\mathrm{DM}}', r'\mathrm{g}~\mathrm{cm}^{-3}']),
'elec_cond': np.array([r'\sigma', r'\mathrm{s}^{-1}']),
'equ_bfield': np.array([r'B_{\mathrm{equ}}', r'\mathrm{G}']),
'equ_bfield_decay': np.array([r'B^{*}_{\mathrm{equ}}', r'\mathrm{G}']),
'hydro_rn': np.array([r'\mathrm{Re}', '-']),
'hydro_rn_decay': np.array([r'\mathrm{Re}^{*}', '-']),
'injection_scale': np.array([r'L_0', r'\mathrm{kpc}']),
'injection_scale_decay': np.array([r'L^{*}_0', r'\mathrm{cm}']),
'kin_visc': np.array([r'\mu_{\parallel}', r'\mathrm{cm}^2~\mathrm{s}^{-1}']),
'mag_diff': np.array([r'\eta', r'\mathrm{cm}^2~\mathrm{s}^{-1}']),
'mag_rn': np.array([r'\mathrm{Rm}', '-']),
'mag_rn_decay': np.array([r'\mathrm{Rm}^{*}', '-']),
'mass': np.array([r'M', r'\mathrm{g}']),
'mfp_i': np.array([r'\lambda_{\mathrm{mfp}, \mathrm{i}}', r'\mathrm{cm}']),
'particle_density': np.array([r'n_e', r'\mathrm{cm}^{-3}']),
'redshift': np.array([r'z', '-']),
'rm_bequ': np.array([r'\sigma_{\mathrm{RM}}(B_{\mathrm{equ}})', r'\mathrm{rad}~\mathrm{m}^{-2}']),
'rm_bequ_decay': np.array([r'\sigma_{\mathrm{RM}}(B^{*}_{\mathrm{equ}})', r'\mathrm{rad}~\mathrm{m}^{-2}']),
'r_range': np.array([r'r', r'\mathrm{cm}']),
'r_range_DM': np.array([r'r_{\mathrm{DM}}', r'\mathrm{cm}']),
'scale_radius': np.array([r'r_s', r'\mathrm{cm}']),
'temperature': np.array([r'T', r'\mathrm{K}']),
'thermvel_ion': np.array([r'v_{\mathrm{th},\mathrm{i}}', r'\mathrm{cm}~\mathrm{s}^{-1}']),
'thermvel_el': np.array([r'v_{\mathrm{th},\mathrm{e}}', r'\mathrm{cm}~\mathrm{s}^{-1}']),
'velocity': np.array([r'v_{\mathrm{turb}}', r'\mathrm{km}~\mathrm{s}^{-1}']),
'velocity_decay': np.array([r'v^{*}_{\mathrm{turb}}', r'\mathrm{km}~\mathrm{s}^{-1}']),
'virial_radius': np.array([r'r_{200}', r'\mathrm{cm}'])
}
