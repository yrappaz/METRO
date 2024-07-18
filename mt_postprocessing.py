import matplotlib.pyplot as plt
import numpy as np

from packages import *
from scipy import integrate
from merger_trees_methods import *
from rotation_measure import *
from external_methods import *
from power_spectrum_methods import *
from statistics_figures import *
import pandas as pd

from figures_single_tree import *
from figures_multiple_trees import *
from methods_3D import *
from power_spectrum_methods import *
from scipy.ndimage import gaussian_filter
from figures_misc import *
from statistics import *
from physical_units import *
from scipy.optimize import curve_fit
from mts_data_analysis import *


def compute_plasma_evolution(mergertree_params):

    log_mclust = mergertree_params['log_mclust']
    log_mres = mergertree_params['log_mres']
    zmax = mergertree_params['zmax']
    zres = mergertree_params['zres']
    nb_real = mergertree_params['nb_real']

    path_trees = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    dir_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/avg_plasma_quantities/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    isdir = os.path.isdir(dir_to_store)
    if isdir == False:
        os.mkdir(dir_to_store)

    not_to_treat = np.array(['redshift', 'rm_bequ', 'rm_bequ_decay', 'r_range', 'r_range_DM'])

    for it in all_plasma_quantities.items():

        name = it[0]
        # if name != 'redshift' or name != 'rm_bequ' or name != 'rm_bequ_decay' or name != 'r_range' or name != 'r_range_DM':
        if np.isin(name,not_to_treat) == False:
            ref_z = np.load(path_trees+f'tree_1/redshift.npy')

            all_trees_rm = np.empty((nb_real, len(ref_z)))

            for i in range(1, nb_real+1):
                # print(i)

                path_data = path_trees+f'tree_{i}/'+name+'.npy'
                data = np.load(path_data, allow_pickle = True)
                z = np.load(path_trees+f'tree_{i}/redshift.npy')

                rms_tree_qty = np.array([])
                for j in range(0, len(z)):
                    if isinstance(data[j], float) == False:
                        rms_tree_qty = np.append(rms_tree_qty, calculate_rms(data[j]))
                    else:
                        rms_tree_qty = np.append(rms_tree_qty, data[j])

                all_trees_rm[i-1] = rms_tree_qty



            np.save(dir_to_store+name+'_all_trees_rms.npy', all_trees_rm)
            np.save(dir_to_store+'redshift.npy', z)


            skewnorm_qty = np.array([])

            for k in range(0, len(z)):

                if len(np.unique(all_trees_rm[:,k])) == 1:
                    skewnorm_qty = np.append(skewnorm_qty, np.unique(all_trees_rm[:,k]))
                else:
                    fit_params_rm = skewnorm.fit(all_trees_rm[:,k])
                    median_rm = skewnorm.mean(*fit_params_rm)
                    skewnorm_qty = np.append(skewnorm_qty, median_rm)

                print(k)

            np.save(dir_to_store+name+'_skewnorm.npy', skewnorm_qty)

            # print(f'{name} done!')
        else:
            continue

def compute_RM_statistics(mergertree_params):

    log_mclust = mergertree_params['log_mclust']
    log_mres = mergertree_params['log_mres']
    zmax = mergertree_params['zmax']
    zres = mergertree_params['zres']
    nb_real = mergertree_params['nb_real']

    path_trees = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    dir_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/RM_analysis/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'

    isdir = os.path.isdir(dir_to_store)
    if isdir == False:
        os.mkdir(dir_to_store)

    ref_z = np.load(path_trees + f'tree_1/redshift.npy')

    all_data = np.empty((nb_real), dtype = np.ndarray)
    all_masses = np.empty((nb_real, len(ref_z)), dtype = float)
    #
    for i in range(1, nb_real + 1):

        path_data = path_trees + f'tree_{i}/rm_bequ.npy'
        data = np.load(path_data, allow_pickle=True)
        z = np.load(path_trees + f'tree_{i}/redshift.npy')
        M = np.load(path_trees + f'tree_{i}/mass.npy')
        rvir = np.load(path_trees + f'tree_{i}/virial_radius.npy')

        all_masses[i-1] = M

        lz = len(z)

        all_data_single_tree = np.empty((lz), dtype = dict)
        for j in range(0, lz):

            temp_dic = {'z': z[j], 'mass': M[j]}

            data_to_treat = data[j][0]
            sup_size = data_to_treat.shape[0]

            eff_sizes = np.array([])
            eff_RM = np.array([])

            for s in range(2, sup_size+1):

                ratio_rvir = s/sup_size
                eff_sizes = np.append(eff_sizes, ratio_rvir)
                all_subm = extract_submatrices(data_to_treat, s)
                eff_RM = np.append(eff_RM, np.mean(calculate_std_array_of_matrices(all_subm, all_subm.shape)))

            temp_dic.update({'size_to_rv': eff_sizes, 'mean_rm': eff_RM})
            all_data_single_tree[j] = temp_dic

        all_data[i-1] = all_data_single_tree

        print(f'{i} done')

    # np.save(dir_to_store+'RM_stdev.npy', all_data)

    # Computing the averaged profiles for each size ratio
    # ---------------------------------------------------------------
    data = np.load(dir_to_store + 'RM_stdev.npy', allow_pickle=True)

    nb_trees = len(data)

    ZREF = np.array([])
    for i in range(0, len(data[0])):
        ZREF = np.append(ZREF, data[0][i]['z'])

    ratio_ref = data[0][0]['size_to_rv']

    all_rm_avg = np.empty((len(ratio_ref), len(ZREF)), dtype = np.ndarray)
    all_rm_skewnorm = np.empty((len(ratio_ref), len(ZREF)), dtype = np.ndarray)
    err_sn_down = np.empty((len(ratio_ref), len(ZREF)), dtype = np.ndarray)
    err_sn_up = np.empty((len(ratio_ref), len(ZREF)), dtype = np.ndarray)
    all_rm_skewnorm_std = np.empty((len(ratio_ref), len(ZREF)), dtype = np.ndarray)

    for i in range(0, len(ZREF)):

        for k in range(0, len(ratio_ref)):

            temp_data = np.array([])
            for j in range(0, nb_trees):

                temp_data = np.append(temp_data,data[j][i]['mean_rm'][k])

            fit_params_rm = skewnorm.fit(temp_data)
            median_rm = skewnorm.median(*fit_params_rm)
            interval = skewnorm.interval(0.9, *fit_params_rm)
            std = skewnorm.std(*fit_params_rm)

            all_rm_avg[k,i] = np.mean(temp_data)
            all_rm_skewnorm[k,i] = median_rm
            err_sn_down[k,i] = interval[0]
            err_sn_up[k,i] = interval[1]
            all_rm_skewnorm_std[k,i] = std

        # print(i)

        # print(f'{i} done')
    np.save(dir_to_store+'AVG_masses.npy', np.mean(all_masses, axis = 0))
    np.save(dir_to_store+'avg_RM_all_trees_ratios.npy', all_rm_avg)
    np.save(dir_to_store+'eff_z.npy', ZREF)
    np.save(dir_to_store+'ratio_to_size.npy', ratio_ref)
    np.save(dir_to_store + 'avg_RM_skewnorm.npy', all_rm_skewnorm)
    np.save(dir_to_store + 'errdown_RM_skewnorm.npy', err_sn_down)
    np.save(dir_to_store + 'errup_RM_skewnorm.npy', err_sn_up)
    np.save(dir_to_store + 'all_rm_skewnorm_std.npy', all_rm_skewnorm_std)
    #
    # ---------------------------------------------------------------

def compute_RM_radial_averaged_profiles(mergertree_params):

    log_mclust = mergertree_params['log_mclust']
    log_mres = mergertree_params['log_mres']
    zmax = mergertree_params['zmax']
    zres = mergertree_params['zres']
    nb_real = mergertree_params['nb_real']

    path_trees = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    dir_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/RM_analysis/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'

    isdir = os.path.isdir(dir_to_store)
    if isdir == False:
        os.mkdir(dir_to_store)

    zref = np.load(path_trees+'tree_1/redshift.npy')
    data_ref = np.load(path_trees + f'tree_1/rm_bequ.npy', allow_pickle=True)
    rvir_ref = np.load(path_trees + f'tree_1/virial_radius.npy')
    radial_avg_ref = get_radial_average_from_matrix(data_ref[0][0], rvir_ref[0], True)[1]
    lrad = len(radial_avg_ref)

    # all_avg_radial_rm = np.empty((len(zref), len(radial_avg_ref)), dtype = float)
    all_avg_radial_skewnorm = np.empty((len(zref), len(radial_avg_ref)), dtype = float)
    all_avg_radial_skewnorm_abs = np.empty((len(zref), len(radial_avg_ref)), dtype = float)

    for i in range(0, len(zref)):

        temp_data = np.empty((nb_real, lrad), dtype = float)
        temp_data_abs = np.empty((nb_real, lrad), dtype = float)

        for j in range(1, nb_real+1):

            data = np.load(path_trees + f'tree_{j}/rm_bequ.npy', allow_pickle=True)
            rvir = np.load(path_trees + f'tree_{j}/virial_radius.npy')

            radial_avg_abs = get_radial_average_from_matrix(data[i][0], rvir[i], True)
            radial_avg = get_radial_average_from_matrix(data[i][0], rvir[i], False)
            temp_data[j-1] = radial_avg[1]
            temp_data_abs[j-1] = radial_avg_abs[1]

        #
        #
        # rms_radavg = np.sqrt((1/nb_real)*np.sum(temp_data**2, axis = 0))
        # all_avg_radial_rm[i] = rms_radavg

        skewnorm_avg = np.array([])
        skewnorm_avg_abs = np.array([])

        for m in range(0, lrad):
            fit_params_rm = skewnorm.fit(temp_data[:,m])
            median_rm = skewnorm.mean(*fit_params_rm)
            skewnorm_avg = np.append(skewnorm_avg, median_rm)

            fit_params_rm_abs = skewnorm.fit(temp_data_abs[:, m])
            median_rm_abs = skewnorm.mean(*fit_params_rm_abs)
            skewnorm_avg_abs = np.append(skewnorm_avg_abs, median_rm_abs)

        all_avg_radial_skewnorm[i] = skewnorm_avg
        all_avg_radial_skewnorm_abs[i] = skewnorm_avg_abs

        print(f'{i}')
    #
    # np.save(dir_to_store+'rm_radial_rms.npy', all_avg_radial_rm)
    np.save(dir_to_store+'rm_radial_skewnorm_abs.npy', all_avg_radial_skewnorm_abs)
    np.save(dir_to_store+'rm_radial_skewnorm_noabs.npy', all_avg_radial_skewnorm)

def compute_RM_power_spectrum_slopes(mergertree_params):

    log_mclust = mergertree_params['log_mclust']
    log_mres = mergertree_params['log_mres']
    zmax = mergertree_params['zmax']
    zres = mergertree_params['zres']
    nb_real = mergertree_params['nb_real']

    path_ref = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_1/'
    z_ref = np.load(path_ref + 'redshift.npy')

    all_bfit = np.empty((len(z_ref), nb_real), dtype=float)
    all_kmax = np.empty((len(z_ref), nb_real), dtype=float)

    for j in range(0, len(z_ref)):

        for i in range(1, nb_real + 1):
            path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{i}/'

            raw_data = np.load(path + 'rm_bequ.npy', allow_pickle=True)
            z = np.load(path + 'redshift.npy')

            data_to_fit = raw_data[j][0]
            kmax, bfit = fit_power_law_rm_power_spectrum(data_to_fit)

            all_bfit[j, i - 1] = bfit
            all_kmax[j, i - 1] = kmax

        # print(f'{j}')

    path_to_save = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/RM_analysis/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'

    isdir = os.path.isdir(path_to_save)
    if isdir == False:
        os.mkdir(path_to_save)

    np.save(path_to_save + 'bfit_params.npy', all_bfit)
    np.save(path_to_save + 'kmax_params.npy', all_kmax)

def compute_ne_bequ_RM_power_spectrum_slopes(index, mult_trees_figures_params):

    def lin_fit(x, a, b):
        return a * x + b

    log_mclust = mult_trees_figures_params['array_log_mclust'][index]
    log_mres = mult_trees_figures_params['array_log_mres'][index]
    zmax = mult_trees_figures_params['array_zmax'][0]
    zres = mult_trees_figures_params['zres']
    nb_real = mult_trees_figures_params['nb_real']
    zlim = mult_trees_figures_params['z_lim_eff']
    # mult_trees_figures_params

    path_ref = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_1/'
    z = np.load(path_ref + 'redshift.npy')
    ind_eff = np.where(z<=zlim)[0]
    z_ref = np.take(z, ind_eff)

    all_slopes_rm = np.empty((len(z_ref), nb_real), dtype=float)
    all_slopes_n = np.empty((len(z_ref), nb_real), dtype=float)
    all_slopes_bequ = np.empty((len(z_ref), nb_real), dtype=float)

    all_shift_rm = np.empty((len(z_ref), nb_real), dtype=float)
    all_shift_n = np.empty((len(z_ref), nb_real), dtype=float)
    all_shift_bequ = np.empty((len(z_ref), nb_real), dtype=float)

    for j in range(0, len(z_ref)):

        for i in range(1, nb_real + 1):
            path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{i}/'

            z = np.load(path+'redshift.npy')
            rm = np.load(path+'rm_bequ.npy', allow_pickle = True)
            n = np.load(path+'particle_density.npy')
            bequ = np.load(path+'equ_bfield.npy')
            inj_scale = np.load(path+'injection_scale.npy')
            rvir = np.load(path+'virial_radius.npy')
            r_range = np.load(path+'r_range.npy')

            # Calculating the 2D power spectrum of the rotation measure
            ps_data_rm = get_spectrum(rm[j][0])
            k_rm, ptot_rm = ps_data_rm['k'],ps_data_rm['P_tot']
            # Calculating the 3D power spectrum of the electron density mapped onto a 3D matrix
            n_3d = map_radial_distribution_onto_3d_matrix(inj_scale[j], rvir[j], n[j])[1]
            ps_data_n = get_spectrum(n_3d)
            k_n, ptot_n = ps_data_n['k'],ps_data_n['P_tot']
            # Calculating the 3D power spectrum of the equipartition magnetic field whose each component is mapped onto a 3D matrix
            bequ_3d = np.empty((3, *n_3d.shape), dtype = float)
            bequ_data = create_3d_magnetic_field_based_on_radial_distrib(inj_scale[j], r_range[j], rvir[j], bequ[j], bequ[j], 1)
            bx,by,bz = bequ_data[1],bequ_data[2],bequ_data[3]
            bequ_3d[0],bequ_3d[1],bequ_3d[2] = bx,by,bz
            ps_data_bequ = get_spectrum(bequ_3d, 3)
            k_bequ, ptot_bequ = ps_data_bequ['k'],ps_data_bequ['P_tot']

            # Fitting the different power spectrum with a linear law
            # ---------- rotation measure ----------
            ind_max_rm = np.argmax(ptot_rm)
            k_rm_fit = k_rm[1:ind_max_rm+1]/np.max(k_rm)
            ptot_rm_fit = ptot_rm[1:ind_max_rm+1]/np.max(ptot_rm[1:ind_max_rm+1])
            popt_rm, pcov_rm = curve_fit(lin_fit, np.log10(k_rm_fit), np.log10(ptot_rm_fit))
            # ---------- 3D equipartition magnetic field ----------
            ind_max_bequ = np.argmax(ptot_bequ)
            k_bequ_fit = k_rm[1:ind_max_bequ+1]/np.max(k_bequ)
            ptot_bequ_fit = ptot_bequ[1:ind_max_bequ+1]/np.max(ptot_bequ[1:ind_max_bequ+1])
            popt_bequ, pcov_bequ = curve_fit(lin_fit, np.log10(k_bequ_fit), np.log10(ptot_bequ_fit))
            # ---------- electron density ----------
            ind_max_n = np.argmax(ptot_n)
            k_n_fit = k_rm[ind_max_n:]/np.max(k_n)
            ptot_n_fit = ptot_n[ind_max_n:]/np.max(ptot_n[ind_max_n:])
            popt_n, pcov_n = curve_fit(lin_fit, np.log10(k_n_fit), np.log10(ptot_n_fit))

            all_slopes_rm[j,i-1] = popt_rm[0]
            all_slopes_n[j,i-1] = popt_n[0]
            all_slopes_bequ[j,i-1] = popt_bequ[0]

            all_shift_rm[j,i-1] = popt_rm[1]
            all_shift_n[j,i-1] = popt_n[1]
            all_shift_bequ[j,i-1] = popt_bequ[1]

        print(f'{j+1}/{len(z_ref)}')

    path_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/RM_analysis/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/power_spectrum_slopes/'

    isdir = os.path.isdir(path_to_store)
    if isdir == False:
        os.mkdir(path_to_store)

    np.save(path_to_store + 'k.npy', k_n)

    np.save(path_to_store+'all_slopes_rm.npy', all_slopes_rm)
    np.save(path_to_store+'all_slopes_n.npy', all_slopes_n)
    np.save(path_to_store+'all_slopes_bequ.npy', all_slopes_bequ)

    np.save(path_to_store+'all_shift_rm.npy', all_shift_rm)
    np.save(path_to_store+'all_shift_n.npy', all_shift_n)
    np.save(path_to_store+'all_shift_bequ.npy', all_shift_bequ)

    all_median_rm = np.array([])
    all_std_rm = np.array([])
    all_median_bequ = np.array([])
    all_std_bequ = np.array([])
    all_median_n  = np.array([])
    all_std_n = np.array([])

    for j in range(0, len(z_ref)):

        fit_params_rm = skewnorm.fit(all_slopes_rm[j])
        median_rm = skewnorm.mean(*fit_params_rm)
        std_rm = skewnorm.std(*fit_params_rm)

        all_median_rm = np.append(all_median_rm, median_rm)
        all_std_rm = np.append(all_std_rm, std_rm)
        # --------------------
        fit_params_bequ = skewnorm.fit(all_slopes_bequ[j])
        median_bequ = skewnorm.mean(*fit_params_bequ)
        std_bequ = skewnorm.std(*fit_params_bequ)

        all_median_bequ = np.append(all_median_bequ, median_bequ)
        all_std_bequ = np.append(all_std_bequ, std_bequ)
        # --------------------
        fit_params_n = skewnorm.fit(all_slopes_n[j])
        median_n = skewnorm.mean(*fit_params_n)
        std_n = skewnorm.std(*fit_params_n)

        all_median_n = np.append(all_median_n, median_n)
        all_std_n = np.append(all_std_n, std_n)

    np.save(path_to_store+'skewn_median_rm.npy', all_median_rm)
    np.save(path_to_store+'skewn_median_bequ.npy', all_median_bequ)
    np.save(path_to_store+'skewn_median_n.npy', all_median_n)

    np.save(path_to_store+'skewn_std_rm.npy', all_std_rm)
    np.save(path_to_store+'skewn_std_bequ.npy', all_std_bequ)
    np.save(path_to_store+'skewn_std_n.npy', all_std_n)

    np.save(path_to_store+'eff_z.npy', z_ref)

def compute_correspondance_between_spectrum(index, mult_trees_figures_params):

    def lin_fit(x, a, b):
        return a * x + b

    def fit(X, a, b):
        lk, lpbequ, lpn = X
        return a + lpn + lpbequ + b * lk

    log_mclust = mult_trees_figures_params['array_log_mclust'][index]
    log_mres = mult_trees_figures_params['array_log_mres'][index]
    zmax = mult_trees_figures_params['array_zmax'][0]
    zres = mult_trees_figures_params['zres']
    nb_real = mult_trees_figures_params['nb_real']
    zlim = mult_trees_figures_params['z_lim_eff']

    path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/RM_analysis/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/power_spectrum_slopes/'

    k = np.load(path + 'k.npy')

    all_slopes_rm = np.load(path + 'all_slopes_rm.npy')
    all_slopes_bequ = np.load(path + 'all_slopes_bequ.npy')
    all_slopes_n = np.load(path + 'all_slopes_n.npy')

    all_shifts_rm = np.load(path + 'all_shift_rm.npy')
    all_shifts_bequ = np.load(path + 'all_shift_bequ.npy')
    all_shifts_n = np.load(path + 'all_shift_n.npy')

    all_fit_a = np.empty((all_slopes_rm.shape[0], all_slopes_rm.shape[1]), dtype = 'float')
    all_fit_b = np.empty((all_slopes_rm.shape[0], all_slopes_rm.shape[1]), dtype = 'float')
    all_fit_a_stdev = np.empty((all_slopes_rm.shape[0], all_slopes_rm.shape[1]), dtype='float')
    all_fit_b_stdev = np.empty((all_slopes_rm.shape[0], all_slopes_rm.shape[1]), dtype='float')



    for i in range(0, all_slopes_rm.shape[0]):
        for j in range(0, all_slopes_rm.shape[1]):
            k_ref = np.log10(k[1:] / np.max(k[1:]))
            p_rm_ref = lin_fit(k_ref, all_slopes_rm[i, j], all_shifts_rm[i, j])
            p_bequ_ref = lin_fit(k_ref, all_slopes_bequ[i, j], all_shifts_bequ[i, j])
            p_n_ref = lin_fit(k_ref, all_slopes_n[i, j], all_shifts_n[i, j])

            X = (k_ref, p_bequ_ref, p_n_ref)
            popt, pcov = curve_fit(fit, X, p_rm_ref)
            err = np.sqrt(np.diag(pcov))

            all_fit_a[i,j] = popt[0]
            all_fit_b[i,j] = popt[1]
            all_fit_a_stdev[i,j] = err[0]
            all_fit_b_stdev[i,j] = err[1]

        print(f'{i}/{all_slopes_rm.shape[0]}')

    np.save(path+'relation_spectrum_fit_param1.npy', all_fit_a)
    np.save(path+'relation_spectrum_fit_param2.npy', all_fit_b)
    np.save(path+'relation_spectrum_fit_param2_stdev.npy', all_fit_b_stdev)
    np.save(path+'relation_spectrum_fit_param2_stdev.npy', all_fit_b_stdev)

    all_median_a = np.array([])
    all_std_a = np.array([])

    all_median_b = np.array([])
    all_std_b = np.array([])

    for i in range(0, all_slopes_rm.shape[0]):
        fit_params_a = skewnorm.fit(all_fit_a[i])
        median_a = skewnorm.mean(*fit_params_a)
        std_a = skewnorm.std(*fit_params_a)

        all_median_a = np.append(all_median_a, median_a)
        all_std_a = np.append(all_std_a, std_a)

        fit_params_b = skewnorm.fit(all_fit_b[i])
        median_b = skewnorm.mean(*fit_params_b)
        std_b = skewnorm.std(*fit_params_b)

        all_median_b = np.append(all_median_b, median_b)
        all_std_b = np.append(all_std_b, std_b)

    np.save(path + 'relation_spectrum_fit_param1_skewnorm_median.npy', all_median_a)
    np.save(path + 'relation_spectrum_fit_param1_skewnorm_std.npy', all_std_a)
    np.save(path + 'relation_spectrum_fit_param2_skewnorm_median.npy', all_median_b)
    np.save(path + 'relation_spectrum_fit_param2_skewnorm_std.npy', all_std_b)

def compute_averaged_profiles(index, qties_averaged, mult_trees_figures_params):

    log_mclust = mult_trees_figures_params['array_log_mclust'][index]
    log_mres = mult_trees_figures_params['array_log_mres'][index]
    zmax = mult_trees_figures_params['array_zmax'][0]
    zres = mult_trees_figures_params['zres']
    nb_real = mult_trees_figures_params['nb_real']
    zlim = mult_trees_figures_params['z_lim_eff']
    resolution = mult_trees_figures_params['resolution']


    path_ref = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_1/'
    z_ref = np.load(path_ref+'redshift.npy')
    eff_z = np.take(z_ref, np.where(z_ref<=zlim)[0])

    AVG_DATA = {}
    for name in qties_averaged:
        AVG_DATA[name] = np.empty((len(eff_z), resolution), dtype=float)


    for j in range(0, len(eff_z)):

        data_to_fit = {}
        for name in qties_averaged:
            data_to_fit[name] = np.empty((nb_real, resolution), dtype = float)

        for i in range(1, nb_real + 1):

            path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{i}/'
            z = np.load(path+'redshift.npy')

            for name in qties_averaged:
                data = np.load(path+name+'.npy', allow_pickle = True)[j]
                data_to_fit[name][i-1] = data


        for name in qties_averaged:
            for k in range(0, resolution):
                # mu, sigma = scipy.stats.norm.fit(data_to_fit[name][:,k])
                fit_params_rm = skewnorm.fit(data_to_fit[name][:,k])
                median_rm = skewnorm.mean(*fit_params_rm)
                AVG_DATA[name][j,k] = median_rm

        print(f'{j+1}/{len(eff_z)} done')
    #
    dir_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/averaged_profiles/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    isdir = os.path.isdir(dir_to_store)
    if isdir == False:
        os.mkdir(dir_to_store)
    #
    for name in qties_averaged:
        np.save(dir_to_store+'avg_profiles_'+name+'.npy', AVG_DATA[name])

    np.save(dir_to_store+'eff_z.npy', eff_z)

def compute_RM_stdev(index, mult_trees_figures_params):

    log_mclust = mult_trees_figures_params['array_log_mclust'][index]
    log_mres = mult_trees_figures_params['array_log_mres'][index]
    zmax = mult_trees_figures_params['array_zmax'][0]
    zres = mult_trees_figures_params['zres']
    nb_real = mult_trees_figures_params['nb_real']
    zlim = mult_trees_figures_params['z_lim_eff']

    def gaussian(X, C, X_mean, sigma):
        return C * exp(-(X - X_mean) ** 2 / (2 * sigma ** 2))

    path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_1/'
    zref = np.load(path+'redshift.npy')
    indices_to_treat = np.where(zref<=zlim)[0][0::20]
    z_to_treat = np.take(zref,indices_to_treat)

    ALL_DATA = np.empty((len(indices_to_treat)), dtype = dict)

    for g in range(0, len(indices_to_treat)):

        index_to_treat = indices_to_treat[g]

        sub_sizes = np.array([2, ])
        all_data = np.empty((len(sub_sizes)), dtype=dict)


        for p in range(0, len(sub_sizes)):

            s = sub_sizes[p]

            # determining the reference array of ratio
            # ------------------------------------------------------------------------------
            radius_ref = np.array([])
            path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_1/'

            rvir = np.load(path + 'virial_radius.npy')[index_to_treat]
            rm = np.load(path + 'rm_bequ.npy', allow_pickle=True)[index_to_treat][0]

            l = rm.shape[0]
            center_x = int(rm.shape[0] / 2)
            center_y = int(rm.shape[0] / 2)
            size_cell = (2 * rvir) / l

            ratio_ref = (size_cell * s) / rvir
            count = 1
            for i in range(center_x, rm.shape[0] - s):
                radius_ref = np.append(radius_ref, (count * size_cell) / rvir)
                count += 1
            # ------------------------------------------------------------------------------

            data_rm_mean = np.empty((nb_real, len(radius_ref)), dtype=float)
            data_rm_std = np.empty((nb_real, len(radius_ref)), dtype=float)

            for k in range(1, nb_real + 1):
                path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{k}/'

                n = np.load(path + 'particle_density.npy')[index_to_treat]
                rvir = np.load(path + 'virial_radius.npy')[index_to_treat]
                rm = np.load(path + 'rm_bequ.npy', allow_pickle=True)[index_to_treat][0]

                l = rm.shape[0]

                center_x = int(rm.shape[0] / 2)
                center_y = int(rm.shape[0] / 2)
                size_cell = (2 * rvir) / l

                ind_m = 0
                for m in range(center_x, rm.shape[0] - s):
                    ycoord = int(rm.shape[0] / 2)
                    sub_m = rm[ycoord - int(s / 2):ycoord + int(s / 2), m - int(s / 2):m + int(s / 2)]

                    sub_m_to_fit = sub_m.flatten()
                    mu, sigma = scipy.stats.norm.fit(sub_m_to_fit)

                    data_rm_mean[k - 1, ind_m] = mu
                    data_rm_std[k - 1, ind_m] = sigma
                    ind_m += 1

            # calculating the skew-normal average of the data
            sn_avg_rm_std = np.array([])
            sn_avg_rm_mean = np.array([])
            for w in range(0, data_rm_mean.shape[1]):

                data_to_fit = data_rm_mean[:, w]
                fit_params_rm = skewnorm.fit(data_to_fit)
                median_rm = skewnorm.mean(*fit_params_rm)
                sn_avg_rm_mean = np.append(sn_avg_rm_mean, median_rm)

                data_to_fit = data_rm_std[:, w]
                fit_params_rm = skewnorm.fit(data_to_fit)
                median_rm = skewnorm.mean(*fit_params_rm)
                sn_avg_rm_std = np.append(sn_avg_rm_std, median_rm)

            all_data[p] = {'size_patch': ratio_ref, 'r_range': radius_ref, 'rm_std': sn_avg_rm_std, 'rm_mean':sn_avg_rm_mean}

        ALL_DATA[g] = {'z': z_to_treat[g], 'data': all_data}

    path_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/RM_analysis/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    np.save(path_to_store+'rm_stdev.npy', ALL_DATA)

def compute_effect_resolution_rm(index_redshift, array_incr_factors, mergertree_params):

    log_mclust = mergertree_params['log_mclust']
    log_mres = mergertree_params['log_mres']
    zmax = mergertree_params['zmax']
    zres = mergertree_params['zres']
    nb_real = mergertree_params['nb_real']

    path_trees = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'

    # **************************************************************************************************************************************************
    # ************************************* Computaion and skew-normal avg of the power spectrum and the radial distribution ***************************
    # **************************************************************************************************************************************************
    all_avg_power_spectrum = np.empty((len(array_incr_factors)), dtype = np.ndarray)
    all_avg_radial_rm = np.empty((len(array_incr_factors)), dtype = np.ndarray)
    all_k = np.empty((len(array_incr_factors)), dtype = np.ndarray)
    # all_r_ratio = np.empty((len(array_incr_factors)), dtype = np.ndarray)


    for k in range(0, len(array_incr_factors)):

        incr_fac = array_incr_factors[k]

        rvir_ref = np.load(path_trees+'tree_1/virial_radius.npy')[index_redshift]
        injection_scale_ref = np.load(path_trees+f'tree_1/injection_scale.npy', allow_pickle = True)[index_redshift]
        r_range_ref = np.load(path_trees+f'tree_1/r_range.npy', allow_pickle = True)[index_redshift]
        equ_bfield_ref = np.load(path_trees+f'tree_1/equ_bfield.npy', allow_pickle = True)[index_redshift]
        bfield_ref = equ_bfield_ref
        particle_density_distrib_ref = np.load(path_trees+f'tree_1/particle_density.npy', allow_pickle = True)[index_redshift]

        rm_ref = compute_rotation_measure_map(injection_scale_ref, r_range_ref, rvir_ref, equ_bfield_ref, bfield_ref, incr_fac, particle_density_distrib_ref, 'x')[0]

        ps_ref = get_spectrum(rm_ref)
        len_ps = len(ps_ref['P_tot'])
        r, rm_r = get_radial_average_from_matrix(rm_ref, rvir_ref, True)

        all_power_spectrum = np.empty((nb_real, len_ps), dtype = float)
        all_rm_radial_avg = np.empty((nb_real, len(rm_r)), dtype = float)

        for i in range(1, nb_real+1):

            rvir = np.load(path_trees+f'tree_{i}/virial_radius.npy', allow_pickle = True)[index_redshift] #loading the virial radius at the index 'index_redshift'
            injection_scale = np.load(path_trees+f'tree_{i}/injection_scale.npy', allow_pickle = True)[index_redshift]
            r_range = np.load(path_trees+f'tree_{i}/r_range.npy', allow_pickle = True)[index_redshift]
            equ_bfield = np.load(path_trees+f'tree_{i}/equ_bfield.npy', allow_pickle = True)[index_redshift]
            bfield = equ_bfield
            particle_density_distrib = np.load(path_trees+f'tree_{i}/particle_density.npy', allow_pickle = True)[index_redshift]

            rm = compute_rotation_measure_map(injection_scale, r_range, rvir, equ_bfield, bfield, incr_fac, particle_density_distrib, 'x')[0]


            # ///////////////////////// Computation (and average) of the power spectrum /////////////////////////
            ps = get_spectrum(rm)

            # ///////////////////////// Computation (and average) of the rm radial distribution /////////////////////////
            r, rm_r = get_radial_average_from_matrix(rm, rvir, True)
            r_ratio = r/rvir

            all_power_spectrum[i-1] = ps['P_tot']
            all_rm_radial_avg[i-1] = rm_r

            print(f'{i}/{nb_real}')



    # ******************* CALCULATING THE SKEW-NORMAL AVG OF ALL THE MERGER TREES ********************

        # print(all_power_spectrum)
        temp_sn_powerspec = np.array([0])
        temp_sn_radial_rm = np.array([])
        for w in range(0, all_power_spectrum.shape[1]):
            if w>0:
                data_to_fit = all_power_spectrum[:,w]
                fit_params_rm = skewnorm.fit(data_to_fit)
                median_rm = skewnorm.mean(*fit_params_rm)
                temp_sn_powerspec = np.append(temp_sn_powerspec, median_rm)

            data_to_fit = all_rm_radial_avg[:, w]
            fit_params_rm = skewnorm.fit(data_to_fit)
            median_rm = skewnorm.mean(*fit_params_rm)
            temp_sn_radial_rm = np.append(temp_sn_radial_rm, median_rm)

        all_avg_power_spectrum[k] = temp_sn_powerspec
        all_avg_radial_rm[k] = temp_sn_radial_rm

        print(f'{k}/{len(array_incr_factors)}')

    # ************************************************************************************************

    dir_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/tests_resolution_effect/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    isdir = os.path.isdir(dir_to_store)
    if isdir == False:
        os.mkdir(dir_to_store)

    np.save(dir_to_store+'all_avg_power_spec.npy', all_avg_power_spectrum)
    np.save(dir_to_store+'all_avg_radial_rm.npy', all_avg_radial_rm)
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************
    # **************************************************************************************************************************************************









        # ///////////////////////// Computation (and average) of the rm stdev /////////////////////////
        # sub_sizes = np.arange(2, (rm.shape[0]/4))
        # all_data = np.empty((len(sub_sizes)), dtype=dict)
        # #
        # for p in range(0, len(sub_sizes)):
        #
        #     s = sub_sizes[p]
        #
        #     # determining the reference array of ratio
        #     # ------------------------------------------------------------------------------
        #     radius_ref = np.array([])
        #     path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_1/'
        #
        #     rvir = np.load(path + 'virial_radius.npy')[index_to_treat]
        #     rm = np.load(path + 'rm_bequ.npy', allow_pickle=True)[index_to_treat][0]
        #
        #     l = rm.shape[0]
        #     center_x = int(rm.shape[0] / 2)
        #     center_y = int(rm.shape[0] / 2)
        #     size_cell = (2 * rvir) / l
        #
        #     ratio_ref = (size_cell * s) / rvir
        #     count = 1
        #     for i in range(center_x, rm.shape[0] - s):
        #         radius_ref = np.append(radius_ref, (count * size_cell) / rvir)
        #         count += 1
        #     # ------------------------------------------------------------------------------
        #
        #     data_rm_mean = np.empty((nb_real, len(radius_ref)), dtype=float)
        #     data_rm_std = np.empty((nb_real, len(radius_ref)), dtype=float)
        #
        #     for k in range(1, nb_real + 1):
        #         path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{k}/'
        #
        #         n = np.load(path + 'particle_density.npy')[index_to_treat]
        #         rvir = np.load(path + 'virial_radius.npy')[index_to_treat]
        #         rm = np.load(path + 'rm_bequ.npy', allow_pickle=True)[index_to_treat][0]
        #
        #         l = rm.shape[0]
        #
        #         center_x = int(rm.shape[0] / 2)
        #         center_y = int(rm.shape[0] / 2)
        #         size_cell = (2 * rvir) / l
        #
        #         ind_m = 0
        #         for m in range(center_x, rm.shape[0] - s):
        #             ycoord = int(rm.shape[0] / 2)
        #             sub_m = rm[ycoord - int(s / 2):ycoord + int(s / 2), m - int(s / 2):m + int(s / 2)]
        #
        #             sub_m_to_fit = sub_m.flatten()
        #             mu, sigma = scipy.stats.norm.fit(sub_m_to_fit)
        #
        #             data_rm_mean[k - 1, ind_m] = mu
        #             data_rm_std[k - 1, ind_m] = sigma
        #             ind_m += 1
        #
        #     # calculating the skew-normal average of the data
        #     sn_avg_rm_std = np.array([])
        #     sn_avg_rm_mean = np.array([])

def compute_radial_central_RM(index, mult_trees_figures_params):
    log_mclust = mult_trees_figures_params['array_log_mclust'][index]
    log_mres = mult_trees_figures_params['array_log_mres'][index]
    zmax = mult_trees_figures_params['array_zmax'][0]
    zres = mult_trees_figures_params['zres']
    nb_real = mult_trees_figures_params['nb_real']
    zlim = mult_trees_figures_params['z_lim_eff']
    resolution = mult_trees_figures_params['resolution']

    path_ref = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_1/'
    z_ref = np.load(path_ref + 'redshift.npy')
    eff_z = np.take(z_ref, np.where(z_ref <= zlim)[0])

    AVG_DATA = {}
    for name in qties_averaged:
        AVG_DATA[name] = np.empty((len(eff_z), resolution), dtype=float)

    for j in range(0, len(eff_z)):

        data_to_fit = {}
        for name in qties_averaged:
            data_to_fit[name] = np.empty((nb_real, resolution), dtype=float)

        for i in range(1, nb_real + 1):

            path = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{i}/'
            z = np.load(path + 'redshift.npy')

            for name in qties_averaged:
                data = np.load(path + name + '.npy', allow_pickle=True)[j]
                data_to_fit[name][i - 1] = data

        for name in qties_averaged:
            for k in range(0, resolution):
                # mu, sigma = scipy.stats.norm.fit(data_to_fit[name][:,k])
                fit_params_rm = skewnorm.fit(data_to_fit[name][:, k])
                median_rm = skewnorm.mean(*fit_params_rm)
                AVG_DATA[name][j, k] = median_rm

        print(f'{j + 1}/{len(eff_z)} done')
    #
    dir_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/averaged_profiles/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    isdir = os.path.isdir(dir_to_store)
    if isdir == False:
        os.mkdir(dir_to_store)
    #
    for name in qties_averaged:
        np.save(dir_to_store + 'avg_profiles_' + name + '.npy', AVG_DATA[name])

    np.save(dir_to_store + 'eff_z.npy', eff_z)


































