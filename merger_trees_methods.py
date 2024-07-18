import os

import numpy as np

from packages import *
from merger_tree_parameters import *
from physical_units import *
from cosmology import *
from external_methods import *

from magnetic_field_evolution import *
from physical_units import *
from methods_3D import *

from scipy import integrate
from multiprocessing import Process
import functools

# Plasma quantities methods
def compute_merging_history_single_tree(dic_parameters):

    zmax = dic_parameters['zmax']
    zres = dic_parameters['zres']
    log_mclust = dic_parameters['log_mclust']
    log_mres = dic_parameters['log_mres']
    single_tree_number = dic_parameters['single_tree_number']

    path_to_merger_data = f'/home/yoan/Desktop/PHD/projects/project_II/data/merger_trees/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'
    # path_to_hdf5_file = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}_data.hdf5'
    #
    # # Test if the hdf5 file exists
    # isfile = os.path.isfile(path_to_hdf5_file)
    # # if note, launch "create_hdf5_file"
    # if isfile == False:
    #     create_hdf5_file(dic_parameters)
    # # open the file to the tree number "single_tree_number"
    # hfile = h5py.File(path_to_hdf5_file, 'a')
    # grp_mt = hfile.require_group(f'tree_{single_tree_number}')

    # constructing the exact name of the file stored in raw mergers data
    # *********************************************************************
    number_string = str(single_tree_number)
    number_of_zeros = 4 - len(number_string)
    filename = 'tree'
    for i in range(0, number_of_zeros):
        filename += '0'
    filename += number_string + '.txt'
    # *********************************************************************

    # open the merger tree raw data file and get store the data
    # *********************************************************************
    file = open(path_to_merger_data + filename, 'r')

    all_lines = file.readlines()
    len_all_lines = len(all_lines)

    data_matrix = np.zeros((len_all_lines,3))

    for i in range(0, len_all_lines):

        temp = np.asarray(all_lines[i].split(' '))
        indices = np.where((temp == '')|(temp == '\n'))
        new_temp = np.delete(temp, indices)

        if len(new_temp)==3:
            data_matrix[i][0] = int(new_temp[0])
            data_matrix[i][1] = float(new_temp[1])
            data_matrix[i][2] = float(new_temp[2])
        else:
            data_matrix[i][1] = new_temp[0]
            data_matrix[i][2] = new_temp[1]

    #_________________________________________________________
    #
    indices_nodes = np.where(data_matrix[:,0] != 0)[0]
    MH = np.empty((len(indices_nodes)),dtype = dict)
    #_________________________________________________________
    #
    for i in range(0, len(data_matrix)-1):
        if data_matrix[i,0] != 0:
            node_this_subhalo = data_matrix[i][0]
            nodes_of_progenitors = np.array([])

            condition = True
            j = i+1

            if data_matrix[j,0] != 0:
                MH[int(data_matrix[i][0])-1] = {'node': int(node_this_subhalo)-1, 'mass':data_matrix[i][1]*solar_masses_to_gr,  'redshift':data_matrix[i][2],  'merged_from_nodes': nodes_of_progenitors}
            else:
                while(condition):
                    if data_matrix[j][0] == 0:
                        # --------- look for the nodes corresponding to this subhalo --------
                        index = np.argwhere((data_matrix[:,0] != 0)&(data_matrix[:,1] == data_matrix[j][1])&(data_matrix[:,2] == data_matrix[j][2]))[0][0]
                        # -------------------------------------------------------------------
                        nodes_of_progenitors = np.append(nodes_of_progenitors, data_matrix[index][0])

                        j = j+1
                    else:
                        condition = False
                MH[int(data_matrix[i][0])-1] = {'node': int(node_this_subhalo)-1, 'mass':data_matrix[i][1]*solar_masses_to_gr,'redshift':data_matrix[i][2],'merged_from_nodes': nodes_of_progenitors-1}
    MH[len(indices_nodes)-1] = {'node': int(data_matrix[len(data_matrix)-1,0])-1, 'mass': data_matrix[len(data_matrix)-1,1]*solar_masses_to_gr, 'redshift':data_matrix[len(data_matrix)-1,2],'merged_from_nodes': np.array([])}
    return MH

    # Storing the data in the hdf5 file
    # for i in range(0, len(MH)):
    #     node = MH[i]['node']
    #     grp_node = grp_mt.require_group(f'node_{node}')
    #
    #     grp_node.attrs['mass'] = MH[i]['mass']
    #     grp_node.attrs['redshift'] = MH[i]['redshift']
    #
    #     mg_fn = MH[i]['merged_from_nodes'].astype(np.float64)
    #
    #
    #     if len(mg_fn)==0:
    #         grp_node.require_dataset(name='merged_from_nodes', dtype=np.float64, shape = (0,), data = np.array([]))
    #     else:
    #         grp_node.require_dataset(name='merged_from_nodes', dtype=np.float64, shape=(len(mg_fn),), data=mg_fn)
    #
    # hfile.close()
def compute_virial_radius_single_tree(dic_parameters):

    H0 = cosmology_params['H0']
    OM_r = cosmology_params['OM_r']
    OM_m = cosmology_params['OM_m']
    OM_k = cosmology_params['OM_k']
    OM_lambda = cosmology_params['OM_lambda']

    data = compute_merging_history_single_tree(dic_parameters)
    L = len(data)

    updated_data = np.copy(data)

    for i in range(0, L):

        s = virial_radius_evolution(updated_data[i]['redshift'], updated_data[i]['mass'], H0, OM_r,OM_m,OM_k,OM_lambda)
        r_range_DM = np.linspace((1e-2)*s,s,mergertree_params['resolution'])
        r_range = np.linspace(0, s, mergertree_params['resolution'])

        injection_scale = s/mergertree_params['alpha_0']

        updated_data[i].update({'virial_radius':s, 'injection_scale': injection_scale, 'r_range':r_range, 'r_range_DM': r_range_DM})

    return updated_data




    #
    #         VR[i] = {"node": i, "redshift": NRM[i]['redshift'], "mass": NRM[i]['mass'], "virial_radius": s,'r_range':r_range, 'r_range_DM': r_range_DM}
def compute_c_param_DM_density_single_tree(dic_parameters):

    data = compute_virial_radius_single_tree(dic_parameters)
    updated_data = np.copy(data)

    # get the unique values redshift
    Z = np.unique([i['redshift'] for i in data])
    L_Z = len(Z)
    L = len(data)

    # Initialisation of the (empty) list of dictionnaries containing the density and information about of each subhalo
    D = np.empty((L), dtype = dict)

    # Initialisation of subhaloes at maximum redshift or with no parents
    for i in range(0, L):

        rv = data[i]['virial_radius']
        r_range = data[i]['r_range_DM']
        Mh = data[i]['mass']

        if len(data[i]['merged_from_nodes']) == 0:

            conc_param = c_for_early_halos(Mh)
            rs = rv/conc_param

            rhos = rho_s(Mh,rv,rs)
            rho_dm = rhos/((r_range/rs)*(1+(r_range/rs))**2)

            D[i] = {'node': data[i]['node'],
                    'redshift': data[i]['redshift'],
                    'mass': Mh,
                    'merged_from_nodes': data[i]['merged_from_nodes'],
                    'r_range_DM': r_range,
                    'r_range': data[i]['r_range'],
                    'virial_radius': data[i]['virial_radius'],
                    'concentration_param': conc_param,
                    'scale_radius': rs,
                    'density_dm': rho_dm}

        else:

            empty_array = np.array([])

            D[i] = {'node': data[i]['node'],
                    'redshift': data[i]['redshift'],
                    'mass': data[i]['mass'],
                    'merged_from_nodes': data[i]['merged_from_nodes'],
                    'r_range_DM': r_range,
                    'r_range': data[i]['r_range'],
                    'virial_radius': data[i]['virial_radius'],
                    'concentration_param': 0,
                    'scale_radius': 0,
                    'density_dm': empty_array}

    # Calculation for the density of all the other halos
    # *****************************************************
    # looping over all decreasing values of redshift
    for i in range(1,L_Z):

        TEMP_M = np.array([])
        TEMP_OD = np.array([])

        # looping over all dictionnaries, and getting the ones corresponding to a given redshift
        for j in range(0,L):

            if D[j]['redshift'] == Z[L_Z-i-1]:

                # Here we catch subhaloes with only ONE parent: they have been evolving and accreting mass, but are not the result of a merger of multiple parents
                if len(D[j]['merged_from_nodes']) == 1:

                    ind = int(D[j]['merged_from_nodes'][0])

                    M_prog = np.array([data[ind]['mass']])
                    rv_prog = np.array([data[ind]['virial_radius']])
                    rs_prog = np.array([D[ind]['scale_radius']])

                    final_M = data[j]['mass']
                    final_rv = data[j]['virial_radius']
                    r_range = data[j]['r_range_DM']

                    new_c, new_rs = compute_c_and_rs_from_energy_conservation(M_prog, rv_prog, rs_prog, final_M, final_rv)

                    rho_DM = rho_s(final_M,final_rv,new_rs)/((r_range/new_rs)*(1+(r_range/new_rs))**2)

                    # Update the value of the dictionnary
                    D[j] = {'node': data[j]['node'],
                    'redshift': data[j]['redshift'],
                    'mass': final_M,
                    'merged_from_nodes': data[j]['merged_from_nodes'],
                    'r_range_DM': r_range,
                    'r_range': data[j]['r_range'],
                    'virial_radius': final_rv,
                    'concentration_param': new_c,
                    'scale_radius': new_rs,
                    'density_dm': rho_dm}

                # Here we catch all the subhalos that result of a merger of multiple parents
                elif len(D[j]['merged_from_nodes']) > 1:

                    # Number of parents that merged together to give this subhalo
                    n_parents = len(D[j]['merged_from_nodes'])

                    array_of_mass = np.array([])
                    array_of_rv = np.array([])
                    array_of_rs = np.array([])

                    for k in range(0, n_parents):

                        # We determine the index that corresponds to each parent
                        ind = int(D[j]['merged_from_nodes'][k])

                        array_of_mass = np.append(array_of_mass, D[ind]['mass'])
                        array_of_rv = np.append(array_of_rv, D[ind]['virial_radius'])
                        array_of_rs = np.append(array_of_rs, D[ind]['scale_radius'])

                    final_M = data[j]['mass']
                    final_rv = data[j]['virial_radius']
                    r_range = data[j]['r_range_DM']

                    new_c, new_rs = compute_c_and_rs_from_energy_conservation(array_of_mass, array_of_rv, array_of_rs, final_M, final_rv)

                    rho_DM = rho_s(final_M,final_rv,new_rs)/((r_range/new_rs)*(1+(r_range/new_rs))**2)

                    # we update the value of the new dictionnary
                    D[j] = {'node': data[j]['node'],
                    'redshift': data[j]['redshift'],
                    'mass': final_M,
                    'merged_from_nodes': data[j]['merged_from_nodes'],
                    'r_range_DM': r_range,
                    'r_range': data[j]['r_range'],
                    'virial_radius': final_rv,
                    'concentration_param': new_c,
                    'scale_radius': new_rs,
                    'density_dm': rho_dm}

        # print('redshift '+str(Z[L_Z-i-1])+' done !')

        # print(f'{i}/{L_Z}')
    for i in range(0, L):

        updated_data[i].update({
            'concentration_param': D[i]['concentration_param'],
            'scale_radius': D[i]['scale_radius'],
            'density_dm': D[i]['density_dm']
        })

    return updated_data
def compute_ICM_gas_temperature_model_single_tree(dic_parameters):

    data = compute_c_param_DM_density_single_tree(dic_parameters)
    updated_data = np.copy(data)

    L = len(data)

    D = np.empty((L), dtype = dict)

    for i in range(0, L):

        rs = data[i]['scale_radius']
        rv = data[i]['virial_radius']
        M = data[i]['mass']
        conc_param = data[i]['concentration_param']

        r = data[i]['r_range']

        rhos = rho_s(M,rv,rs)

        n = (mergertree_params['gamma']-1)**(-1)

        # -----------------------------------------------------------------
        vvir = np.sqrt(G * M / rv)
        Tvir = ((0.5 * mp) / (2 * kb)) * vvir**2
        T0 = Tvir + ((4*math.pi*G*rhos*(rs**2)*0.5*mp)/((1+n)*kb))*(1-np.log(1+rv/rs)/(rv/rs))

        B = B_const(rhos, rs, T0)

        T_profile = T0*(   1-(B/(n+1))*(1-np.log(1+r[1:]/rs)/(r[1:]/rs))      )
        T_profile = np.insert(T_profile, 0, T0)
        # -----------------------------------------------------------------
        dens_integrand = (r[1:]**2)*(1-(B/(n+1))*(1-(np.log(1+r[1:]/rs))/(r[1:]/rs)))**n
        dens_integrand = np.insert(dens_integrand, 0, 0)

        I = integrate.simps(dens_integrand,r)
        rho0 = (mergertree_params['fb']*M)/(4*math.pi*I)

        density_gas = rho0*(1-(B/(n+1))*(1-(np.log(1+r[1:]/rs))/(r[1:]/rs)))**n
        density_gas = np.insert(density_gas, 0, rho0)

        numerical_density = density_gas/(me+mp)

        # # -------- Computation of the turbulent velocity profile ----------
        integral_num = integrate.simps(3*(r**2)*numerical_density*kb*T_profile,r)
        integral_denum = integrate.simps((r**2)*density_gas*(1+r/rv), r)

        v0 = np.sqrt(0.1*integral_num/integral_denum)

        turb_vel = v0*np.sqrt(1+r/rv)

        updated_data[i].update({
            'density_gas': density_gas,
            'temperature': T_profile,
            'particle_density': numerical_density,
            'velocity': turb_vel
        })
        # print(f'{i}/{L}')
    return updated_data
def compute_decaying_turbulence_model_single_tree(dic_parameters):

    data = compute_ICM_gas_temperature_model_single_tree(dic_parameters)
    updated_data = np.copy(data)

    # get the unique values redshift
    Z = np.unique([i['redshift'] for i in data])
    L_Z = len(Z)
    L = len(data)

    # Initialisation of the (empty) list of dictionnaries containing the density and information about of each subhalo
    # D = np.empty((L), dtype = dict)

    # Initialisation of subhaloes at maximum redshift or with no parents
    for i in range(0, L):

        if len(updated_data[i]['merged_from_nodes']) == 0:

            rv = updated_data[i]['virial_radius']
            r = updated_data[i]['r_range']
            numerical_density = updated_data[i]['particle_density']
            T_profile = updated_data[i]['temperature']
            density_gas = updated_data[i]['density_gas']

            # INITIALIZATION OF SUBAHLOES HERE
            integral_num = integrate.simps(3 * (r ** 2) * numerical_density * kb * T_profile, r)
            integral_denum = integrate.simps((r ** 2) * density_gas * (1 + r / rv), r)

            v0 = np.sqrt(0.1 * integral_num / integral_denum)

            turb_vel = v0 * np.sqrt(1 + r / rv)

            dic_temp = {'major_merger': 0,'velocity_decay': turb_vel, 'injection_scale_decay': updated_data[i]['virial_radius']/mergertree_params['alpha_0']}
            updated_data[i].update(dic_temp)
            # print(updated_data[i]['injection_scale'], updated_data[i]['injection_scale_decay'])

        else:

            empty_array = np.array([])
            dic_temp = {'major_merger': 0,'velocity_decay': empty_array, 'injection_scale_decay': 0}
            updated_data[i].update(dic_temp)

    # Calculation for the density of all the other halos
    # *****************************************************
    # looping over all decreasing values of redshift
    for i in range(1,L_Z):

        TEMP_M = np.array([])
        TEMP_OD = np.array([])

        # looping over all dictionnaries, and getting the ones corresponding to a given redshift
        for j in range(0,L):

            if updated_data[j]['redshift'] == Z[L_Z-i-1]:

                # Here we catch subhaloes with only ONE parent: they have been evolving and accreting mass, but are not the result of a merger of multiple parents
                if len(updated_data[j]['merged_from_nodes']) == 1:

                    ind = int(updated_data[j]['merged_from_nodes'][0])

                    zi = updated_data[ind]['redshift']
                    zf = updated_data[j]['redshift']
                    init_r = updated_data[ind]['r_range']
                    init_vturb = updated_data[ind]['velocity_decay']
                    init_L0 = updated_data[ind]['injection_scale_decay']
                    init_density_gas = updated_data[ind]['density_gas']
                    r_range_new = updated_data[j]['r_range']
                    density_gas_new = updated_data[j]['density_gas']
                    temperature_new = updated_data[j]['temperature']
                    particle_density_new = updated_data[j]['particle_density']

                    new_L0, new_velocity = adjust_turbulent_velocity_and_injection_scale_decay(zi, zf, init_r, init_vturb, init_L0, init_density_gas,  r_range_new, density_gas_new, temperature_new, particle_density_new)
                    updated_data[j].update({'major_merger': 0, 'velocity_decay': new_velocity,'injection_scale_decay': new_L0})

                # Here we catch all the subhalos that result of a merger of multiple parents
                elif len(updated_data[j]['merged_from_nodes']) > 1:

                    # Number of parents that merged together to give this subhalo
                    n_parents = len(updated_data[j]['merged_from_nodes'])
                    array_of_mass = np.array([])

                    for k in range(0, n_parents):

                        # We determine the index that corresponds to each parent
                        ind = int(updated_data[j]['merged_from_nodes'][k])
                        array_of_mass = np.append(array_of_mass, updated_data[ind]['mass'])

                    final_M = data[j]['mass']
                    final_rv = data[j]['virial_radius']
                    r_range = data[j]['r_range_DM']

                    mtype = determine_merger_type(final_M, array_of_mass,0.1)


                    if mtype == 'major':

                        rv = data[j]['virial_radius']
                        r = data[j]['r_range']
                        numerical_density = data[j]['particle_density']
                        T_profile = data[j]['temperature']
                        density_gas = data[j]['density_gas']

                        integral_num = integrate.simps(3 * (r ** 2) * numerical_density * kb * T_profile, r)
                        integral_denum = integrate.simps((r ** 2) * density_gas * (1 + r / rv), r)
                        v0 = np.sqrt(0.1 * integral_num / integral_denum)
                        turb_vel = v0 * np.sqrt(1 + r / rv)
                        dic_temp = {'major_merger': 1,'velocity_decay': turb_vel, 'injection_scale_decay': rv/mergertree_params['alpha_0']}
                        updated_data[j].update(dic_temp)
                        # print('************* major merger !******************')
                        # print(rv/mergertree_params['alpha_0'], updated_data[j]['injection_scale'])
                        # print(updated_data[j]['injection_scale'], updated_data[j]['injection_scale_decay'])
                    elif mtype == 'minor':
                        print('bleh')
                            # ind = int(updated_data[j]['merged_from_nodes'][0])

                        # determining the index corresponding to the most massive parent
                        where_massive = np.argwhere(array_of_mass == np.max(array_of_mass))
                        ind = updated_data[j]['merged_from_nodes'][where_massive]

                        zi = updated_data[ind]['redshift']
                        zf = updated_data[j]['redshift']
                        init_r = updated_data[ind]['r_range']
                        init_vturb = updated_data[ind]['velocity_decay']
                        init_L0 = updated_data[ind]['injection_scale_decay']
                        r_range_new = updated_data[j]['r_range']
                        density_gas_new = updated_data[j]['density_gas']
                        temperature_new = updated_data[j]['temperature']
                        particle_density_new = updated_data[j]['particle_density']

                        new_L0, new_veloctiy = adjust_turbulent_velocity_and_injection_scale_decay(zi, zf, init_r, init_vturb, init_L0,
                                                                            r_range_new, density_gas_new, temperature_new,
                                                                            particle_density_new)
                        updated_data[j].update({'major_merger': 0, 'velocity_decay': new_veloctiy,
                                                'injection_scale_decay': new_L0})

    return updated_data
def compute_plasma_quantities_single_tree(dic_parameters):


        # data = compute_ICM_gas_temperature_model_single_tree(dic_parameters)
        data = compute_decaying_turbulence_model_single_tree(dic_parameters)
        updated_data = np.copy(data)

        L = len(data)

        for i in range(0,L):

            temperature = data[i]['temperature']
            part_density = data[i]['particle_density']

            velocity = data[i]['velocity']
            velocity_decay = data[i]['velocity_decay']
            injection_scale = data[i]['injection_scale']
            injection_scale_decay = data[i]['injection_scale_decay']
            #
            gas_density = data[i]['density_gas']

            # THERMAL VELOCITY
            thermvel_ion = np.sqrt((2*kb*temperature)/(mp))
            thermvel_el = np.sqrt((2*kb*temperature)/(me))

            # COLLISION FREQUENCY
            v = np.abs(thermvel_ion-thermvel_el)
            reduced_mass = (me*mp)/(me+mp)

            coulomb_log = np.log(np.sqrt(kb*temperature/(4*math.pi*part_density*e**2)) / ((e**2)/(reduced_mass*v**2)))
            coll_freq = (4/3)*np.sqrt(2)*(math.pi)**(3/2)*(e**4*part_density*coulomb_log/(np.sqrt(me)))*(kb*temperature)**(-3/2)
            coll_freq_ii = 2*(1/math.sqrt(2))*math.sqrt(me/mp)*coll_freq

            mean_free_path_i = thermvel_ion/(coll_freq_ii)

            # KINEMATIC VVISCOSITY
            kin_visc = thermvel_ion*mean_free_path_i

            # CLASSICAL REYNOLDS NUMBER
            hydro_rn = velocity * injection_scale / kin_visc
            hydro_rn_decay = velocity_decay * injection_scale_decay / kin_visc

            # ELECTRICAL CONDUCTIVITY
            elec_cond = (part_density* e ** 2) / (me * coll_freq_ii)

            # MAGNETIC DIFFUSIVITY
            mag_diff = (c ** 2) / (4 * math.pi * elec_cond)

            # MAGNETIC REYNOLDS NUMBER
            mag_rn = velocity*injection_scale/mag_diff
            mag_rn_decay = velocity_decay*injection_scale_decay/mag_diff

            # EQUIPARTITION MAGNETIC FIELD
            equ_bfield = np.sqrt(4 * math.pi * gas_density * velocity ** 2)
            equ_bfield_decay = np.sqrt(4 * math.pi * gas_density * velocity_decay ** 2)

            updated_data[i].update({
            'thermvel_ion': thermvel_ion,
            'thermvel_el': thermvel_el,
            'coll_freq_ii':coll_freq_ii,
            'mfp_i':mean_free_path_i,
            'kin_visc':kin_visc,
            'hydro_rn':hydro_rn,
            'hydro_rn_decay':hydro_rn_decay,
            'elec_cond':elec_cond,
            'mag_diff':mag_diff,
            'mag_rn':mag_rn,
            'mag_rn_decay':mag_rn_decay,
            'equ_bfield':equ_bfield,
            'equ_bfield_decay':equ_bfield_decay
            })

        return updated_data

def create_directories_to_store_data(dic_parameters):

    zmax = dic_parameters['zmax']
    zres = dic_parameters['zres']
    log_mclust = dic_parameters['log_mclust']
    log_mres = dic_parameters['log_mres']
    tree_number = dic_parameters['single_tree_number']
    nb_trees = dic_parameters['nb_real']

    path_dir = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'

    isdir = os.path.isdir(path_dir)
    if isdir == False:
        os.makedirs(path_dir)

    for i in range(1, nb_trees+1):

        path_tree = path_dir+f'tree_{i}/'

        isdir = os.path.isdir(path_tree)
        if isdir == False:
            os.makedirs(path_tree)
def store_data_most_massive_halos_single_tree(dic_parameters):


    zmax = dic_parameters['zmax']
    zres = dic_parameters['zres']
    log_mclust = dic_parameters['log_mclust']
    log_mres = dic_parameters['log_mres']
    tree_number = dic_parameters['single_tree_number']
    resolution = dic_parameters['resolution']


    path_to_store = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/'

    # Computation of plasma quantities for all subhaloes of a given tree
    data = compute_plasma_quantities_single_tree(dic_parameters)
    L = len(data)
    #
    # *******************************************************************
    # Loop over the data and find the most massive halo at each redshift
    # *******************************************************************

    # create an array with all unique values of redshift
    all_z = np.array([])
    for i in range(0, L):
        all_z = np.append(all_z, data[i]['redshift'])
    all_z_unique = np.unique(all_z)

    # create an array of nodes corresponding to most massive halos
    massive_nodes = np.array([])

    for z in all_z_unique:
        temp_masses = np.array([])
        temp_nodes = np.array([])
        for j in range(0, L):
            if data[j]['redshift'] == z:
                temp_masses = np.append(temp_masses, data[j]['mass'])
                temp_nodes = np.append(temp_nodes, data[j]['node'])

        ind_max = np.argmax(temp_masses)
        massive_nodes = np.append(massive_nodes, int(temp_nodes[ind_max]))
    massive_nodes = massive_nodes.astype(int)

    # store all plasma quantities data correesponding to these nodes into a 2D matrix
    list_quantities = {}

    for it in data[0].items():
        if it[0]!='node' and it[0]!='merged_from_nodes':
            if it[0] == 'redshift'\
            or it[0] == 'concentration_param' \
            or it[0] == 'injection_scale' \
            or it[0] == 'injection_scale_decay' \
            or it[0] == 'major_merger' \
            or it[0] == 'virial_radius' \
            or it[0] == 'mass' \
            or it[0] == 'scale_radius':

                list_quantities.update({it[0]: np.empty((len(massive_nodes)), dtype = float)})

            else:
                list_quantities.update({it[0]: np.empty((len(massive_nodes), resolution), dtype = float)})

    # store the list of massive nodes
    np.save(path_to_store+'nodes.npy', massive_nodes)

    # store all data in the matrices in list_quantities
    count = 0
    for ind in massive_nodes:

        for qty in list_quantities:

            list_quantities[qty][count] = data[ind][qty]

        count+=1

    # Print all data in the folder "path_to_store"
    for it in list_quantities.items():
        np.save(path_to_store+it[0]+'.npy', it[1])

    print(f'tree {tree_number} done !')
    data = None
    list_quantities = None

# Magnetic fields and rotation measure maps methods
def compute_magnetic_field_single_tree(dic_parameters, bfield_params):

    zmax = dic_parameters['zmax']
    zres = dic_parameters['zres']
    log_mclust = dic_parameters['log_mclust']
    log_mres = dic_parameters['log_mres']
    tree_number = dic_parameters['single_tree_number']

    b0_init = bfield_params['b0']
    mag_res_fraction = bfield_params['mag_res_fraction']
    z_start = bfield_params['z_start']

    incr_factor = bfield_params['incr_factor']

    path_bf = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/bfields_data/mag_res_frac_1e{int(np.log10(mag_res_fraction))}/b0_{b0_init}/zst_{z_start}/'
    path_rm = path_bf+f'incr_factor_{incr_factor}/'
    path_plasma = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/'

    redshift = np.load(path_plasma+'redshift.npy')
    rvir = np.load(path_plasma+'virial_radius.npy')
    r_range = np.load(path_plasma+'r_range.npy')
    velocity = np.load(path_plasma+'velocity.npy')
    thermvel_ion = np.load(path_plasma+'thermvel_ion.npy')
    equ_bfield = np.load(path_plasma+'equ_bfield.npy')
    density_gas = np.load(path_plasma+'density_gas.npy')
    particle_density = np.load(path_plasma+'particle_density.npy')
    hydro_rn = np.load(path_plasma+'hydro_rn.npy')
    temperature = np.load(path_plasma+'temperature.npy')
    virial_radius = np.load(path_plasma+'virial_radius.npy')
    length = np.load(path_plasma+'injection_scale.npy')

    init_b0_values = b0_init*np.ones(mergertree_params['resolution'])
    # init_b0_values = b0_init*np.random.rand(mergertree_params['resolution'])



    t1 = time.time()
    z_eff, re_eff_L, bf_L, re_eff_M, bf_M, re_eff_U, bf_U = bfield_evolution_from_zstart_array(redshift, velocity,
                                                                                               thermvel_ion, equ_bfield,
                                                                                               density_gas,
                                                                                               particle_density,
                                                                                               hydro_rn, temperature,
                                                                                               virial_radius, length,
                                                                                               mag_res_fraction,
                                                                                               z_start, init_b0_values)
    # t2 = time.time()
    # print(t2-t1)
    # t1 = time.time()
    # z_eff, re_eff_A_decay, bf_A_decay, re_eff_B_decay, bf_B_decay, re_eff_C_decay, bf_C_decay = bfield_evolution_from_zstart_array(redshift, velocity_decay,
    #                                                                                            thermvel_ion, equ_bfield_decay,
    #                                                                                            density_gas,
    #                                                                                            particle_density,
    #                                                                                            hydro_rn_decay, temperature,
    #                                                                                            virial_radius, length_decay,
    #                                                                                            mag_res_fraction,
    #                                                                                            z_start, init_b0_values)

    # t2 = time.time()
    # print(t2 - t1)
    path_data = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/'
    path_to_mag_data = path_data+'bfields_data/'
    isdir = os.path.isdir(path_to_mag_data)
    if isdir == False:
        os.makedirs(path_to_mag_data)

    path_to_res = path_to_mag_data + f'mag_res_frac_1e{int(np.log10(mag_res_fraction))}/'
    isdir = os.path.isdir(path_to_res)
    if isdir == False:
        os.makedirs(path_to_res)

    path_to_b0 = path_to_res + f'b0_{b0_init}/'
    isdir = os.path.isdir(path_to_b0)
    if isdir == False:
        os.makedirs(path_to_b0)

    path_to_zst = path_to_b0 + f'zst_{z_start}/'
    isdir = os.path.isdir(path_to_zst)
    if isdir == False:
        os.makedirs(path_to_zst)

    np.save(path_to_zst + 'z_eff.npy', z_eff)

    np.save(path_to_zst+'bf_L.npy', bf_L)
    np.save(path_to_zst+'bf_M.npy', bf_M)
    np.save(path_to_zst+'bf_U.npy', bf_U)

    np.save(path_to_zst+'re_eff_L.npy', re_eff_L)
    np.save(path_to_zst+'re_eff_M.npy', re_eff_M)
    np.save(path_to_zst+'re_eff_U.npy', re_eff_U)
def compute_rotation_measure_maps_single_tree(dic_parameters, bfield_params):

    zmax = dic_parameters['zmax']
    zres = dic_parameters['zres']
    log_mclust = dic_parameters['log_mclust']
    log_mres = dic_parameters['log_mres']
    tree_number = dic_parameters['single_tree_number']

    zst = bfield_params['z_start']
    mag_frac = bfield_params['mag_res_fraction']
    b0 = bfield_params['b0']
    incr_factor = bfield_params['incr_factor']

    axis = 'x'

    path_to_plasma_data = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/'

    path_to_bfield_data = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/' \
                          f'logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/' \
                          f'tree_{tree_number}/bfields_data/mag_res_frac_1e{int(np.log10(mag_frac))}/b0_{b0}/zst_{zst}/'

    z_eff = np.load(path_to_bfield_data+'z_eff.npy')
    lz = len(z_eff)

    virial_radius = np.load(path_to_plasma_data+'virial_radius.npy')
    r_range = np.load(path_to_plasma_data+'r_range.npy')
    equ_bfield = np.load(path_to_plasma_data+'equ_bfield.npy')
    particle_density = np.load(path_to_plasma_data+'particle_density.npy')
    injection_scale = np.load(path_to_plasma_data+'injection_scale.npy')

    bf_L = np.load(path_to_bfield_data+'bf_L.npy')
    bf_M = np.load(path_to_bfield_data+'bf_M.npy')
    bf_U = np.load(path_to_bfield_data+'bf_U.npy')

    rm_data_L = np.empty((lz), dtype = np.ndarray)
    rm_data_M = np.empty((lz), dtype = np.ndarray)
    rm_data_U = np.empty((lz), dtype = np.ndarray)
    #

    for i in range(0, lz):
        # print(i, z_eff[i])
        # print('----------------------------------------')
        rm_data_L[i] = compute_rotation_measure_map(injection_scale[i], r_range[i], virial_radius[i], equ_bfield[i], bf_L[i], incr_factor, particle_density[i], 'x')
#         print('mod L')
        rm_data_M[i] = compute_rotation_measure_map(injection_scale[i], r_range[i], virial_radius[i], equ_bfield[i], bf_M[i], incr_factor, particle_density[i], 'x')
#         print('mod M')
        rm_data_U[i] = compute_rotation_measure_map(injection_scale[i], r_range[i], virial_radius[i], equ_bfield[i], bf_U[i], incr_factor, particle_density[i], 'x')
#         print('mod U')

        # print(f'{i}/{lz-1} done')
        # print(f'{i}/{lz-1}')

    path_to_incr_dir = path_to_bfield_data+f'incr_factor_{incr_factor}/'

    isdir = os.path.isdir(path_to_incr_dir)
    if isdir == False:
        os.makedirs(path_to_incr_dir)

    np.save(path_to_incr_dir+'rm_L.npy', rm_data_L)
    np.save(path_to_incr_dir+'rm_M.npy', rm_data_M)
    np.save(path_to_incr_dir+'rm_U.npy', rm_data_U)
def compute_rotation_measure_map_equbf_single_tree(dic_parameters):

    zmax = dic_parameters['zmax']
    zres = dic_parameters['zres']
    log_mclust = dic_parameters['log_mclust']
    log_mres = dic_parameters['log_mres']
    tree_number = dic_parameters['single_tree_number']

    axis = 'x'

    path_to_plasma_data = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/'


    redshift = np.load(path_to_plasma_data+'redshift.npy')
    lz = len(redshift)

    virial_radius = np.load(path_to_plasma_data+'virial_radius.npy')
    r_range = np.load(path_to_plasma_data+'r_range.npy')
    equ_bfield = np.load(path_to_plasma_data+'equ_bfield.npy')
    equ_bfield_decay = np.load(path_to_plasma_data+'equ_bfield_decay.npy')
    particle_density = np.load(path_to_plasma_data+'particle_density.npy')
    injection_scale = np.load(path_to_plasma_data+'injection_scale.npy')
    injection_scale_decay = np.load(path_to_plasma_data+'injection_scale_decay.npy')

    rm_data = np.empty((lz), dtype = np.ndarray)
    rm_data_decay = np.empty((lz), dtype = np.ndarray)
    # radial_avg_rm = np.empty((lz), dtype = np.ndarray)
    # radial_avg_rm_decay = np.empty((lz), dtype = np.ndarray)
    # all_range = np.empty((lz), dtype = np.ndarray)

    incr_factor = 1

    for i in range(0, lz):

        rm_data[i] = compute_rotation_measure_map(injection_scale[i], r_range[i], virial_radius[i], equ_bfield[i], equ_bfield[i], incr_factor, particle_density[i], 'x')
        rm_data_decay[i] = compute_rotation_measure_map(injection_scale_decay[i], r_range[i], virial_radius[i], equ_bfield_decay[i], equ_bfield_decay[i], incr_factor, particle_density[i], 'x')
        # all_range = get_radial_average_from_matrix(rm_data[i], virial_radius[i], True)[0]
        # radial_avg_rm[i] = get_radial_average_from_matrix(rm_data[i], virial_radius[i], True)[1]
        # radial_avg_rm_decay[i] = get_radial_average_from_matrix(rm_data_decay[i], virial_radius[i], True)[1]

    np.save(path_to_plasma_data+'rm_bequ.npy', rm_data)
    np.save(path_to_plasma_data+'rm_bequ_decay.npy', rm_data_decay)
    # np.save(path_to_plasma_data+'radial_avg_rm.npy', radial_avg_rm)
    # np.save(path_to_plasma_data+'radial_avg_rm_decay.npy', radial_avg_rm_decay)
    # np.save(path_to_plasma_data+'all_range.npy', all_range)

# Methods for computing multiple merger trees
# ////////////////////////////////////////////////////////////////////////////////
# plasma parameters
def pick_tree_number_and_store_plasma_data(x, dic_parameters):

    copy_params = dic_parameters.copy()
    copy_params['single_tree_number'] = x

    store_data_most_massive_halos_single_tree(copy_params)
    # compute_rotation_measure_map_equbf_single_tree(copy_params)
def store_plasma_data_all_trees(dic_parameters):

    create_directories_to_store_data(dic_parameters)

    # print(multiprocessing.cpu_count())
    data_to_process = np.arange(1, dic_parameters['nb_real'] + 1).tolist()
    pool = multiprocessing.Pool(10)
    partial_F = functools.partial(pick_tree_number_and_store_plasma_data, dic_parameters=dic_parameters)
    results = pool.map(partial_F, data_to_process)
    pool.close()
    pool.join()

# magnetic fields and rotation measure maps
def compute_bfield_data_all_parameters_single_tree(x, dic_parameters, bfield_params):

    copy_params = dic_parameters.copy()
    copy_params['single_tree_number'] = x

    copy_bf_params = bfield_params.copy()

    all_b0 = bfield_params['all_b0']
    all_z_start = bfield_params['all_z_start']
    all_mag_res = bfield_params['all_mag_res_fractions']

    all_bf_operations = len(all_b0)*len(all_z_start)*len(all_mag_res)
    counter = 1

    for b0 in all_b0:
        copy_bf_params['b0'] = b0

        for zst in all_z_start:
            copy_bf_params['z_start'] = zst

            for mag_res in all_mag_res:
                copy_bf_params['mag_res_fraction'] = mag_res

                compute_magnetic_field_single_tree(copy_params, copy_bf_params)
                #
                # print(f'BF {counter}/{all_bf_operations}')
                # counter += 1

    print(f'BF tree {x} done')
def store_bfield_all_trees(dic_parameters, bfield_params):

    data_to_process = np.arange(1, dic_parameters['nb_real'] + 1).tolist()
    pool = multiprocessing.Pool(10)
    partial_F = functools.partial(compute_bfield_data_all_parameters_single_tree, dic_parameters=dic_parameters, bfield_params = bfield_params)
    results = pool.map(partial_F, data_to_process)
    pool.close()
    pool.join()

def compute_RM_data_all_parameters_single_tree(x, dic_parameters):

    copy_params = dic_parameters.copy()
    copy_params['single_tree_number'] = x

    counter = 1

    for fac in copy_params['incr_factors']:
        copy_params['incr_factor'] = fac
        compute_rotation_measure_map_equbf_single_tree(copy_params)

    print(f'RM tree {x} done')
def store_RM_all_trees(dic_parameters):

    data_to_process = np.arange(1, dic_parameters['nb_real'] + 1).tolist()
    pool = multiprocessing.Pool(10)
    partial_F = functools.partial(compute_RM_data_all_parameters_single_tree, dic_parameters=dic_parameters, bfield_params = bfield_params)
    results = pool.map(partial_F, data_to_process)
    pool.close()
    pool.join()
# ////////////////////////////////////////////////////////////////////////////////
#python
# def compute_minimum_equipartition_redshift(dic_parameters, bfield_params):
#
#     zmax = dic_parameters['zmax']
#     zres = dic_parameters['zres']
#     log_mclust = dic_parameters['log_mclust']
#     log_mres = dic_parameters['log_mres']
#     tree_number = dic_parameters['single_tree_number']
#
#     # b0_init = bfield_params['b0']
#     mag_res_fraction = bfield_params['mag_res_fraction']
#     all_zstart = bfield_params['all_z_start']
#     all_b0 = bfield_params['all_b0']
#
#     path_plasma = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/'
#     equ_bfield = np.load(path_plasma+'equ_bfield.npy')
#
#     zmin_equ_L = np.empty((len(all_b0), 2), dtype = float)
#     zmin_equ_M = np.empty((len(all_b0), 2), dtype = float)
#     zmin_equ_U = np.empty((len(all_b0), 2), dtype = float)
#
#     for i in range(0, len(all_b0)):
#
#         bool_equ_L = False
#
#         for zst in all_zstart:
#
#             equ_bfield_rms = calculate_rms(equ_bfield[0])
#             path_bf = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/bfields_data/mag_res_frac_1e{int(np.log10(mag_res_fraction))}/b0_{all_b0[i]}/zst_{zst}/'
#             bf_L = np.load(path_bf+'bf_L.npy')[0]
#             bf_L_rms = calculate_rms(bf_L)
#
#             if bf_L_rms >= 0.1*equ_bfield_rms and bf_L_rms <= 1.1*equ_bfield_rms:
#                 zmin_equ_L[i,:] = np.array([all_b0[i], zst])
#                 bool_equ_L = True
#                 break
#
#         for zst in all_zstart:
#
#             equ_bfield_rms = calculate_rms(equ_bfield[0])
#             path_bf = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/bfields_data/mag_res_frac_1e{int(np.log10(mag_res_fraction))}/b0_{all_b0[i]}/zst_{zst}/'
#             bf_M = np.load(path_bf+'bf_M.npy')[0]
#             bf_M_rms = calculate_rms(bf_M)
#
#             if bf_M_rms >= 0.1*equ_bfield_rms and bf_M_rms <= 1.1*equ_bfield_rms:
#                 zmin_equ_M[i,:] = np.array([all_b0[i], zst])
#                 bool_equ_M = True
#                 break
#
#         for zst in all_zstart:
#
#             equ_bfield_rms = calculate_rms(equ_bfield[0])
#             path_bf = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/bfields_data/mag_res_frac_1e{int(np.log10(mag_res_fraction))}/b0_{all_b0[i]}/zst_{zst}/'
#             bf_U = np.load(path_bf+'bf_U.npy')[0]
#             bf_U_rms = calculate_rms(bf_U)
#
#             if bf_U_rms >= 0.1*equ_bfield_rms and bf_U_rms <= 1.1*equ_bfield_rms:
#                 zmin_equ_U[i,:] = np.array([all_b0[i], zst])
#                 bool_equ_U = True
#                 break
#
#         if bool_equ_L == False:
#             zmin_equ_L[i, :] = np.array([all_b0[i], 0])
#         if bool_equ_M == False:
#             zmin_equ_M[i, :] = np.array([all_b0[i], 0])
#         if bool_equ_U == False:
#             zmin_equ_U[i, :] = np.array([all_b0[i], 0])







    # print(zmin_equ)

def compute_and_store_RM_stdev_all_trees(dic_parameters, bfield_params):

    zmax = mergertree_params['zmax']
    zres = mergertree_params['zres']
    log_mclust = mergertree_params['log_mclust']
    log_mres = mergertree_params['log_mres']
    all_tree_number = mergertree_params['nb_real']

    all_zst = bfield_params['all_z_start']
    mag_frac = bfield_params['mag_res_fraction']
    all_b0 = bfield_params['all_b0']
    incr_factor = bfield_params['incr_factor']

    path_to_mt = f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/txt_data/'

    total_op = len(all_b0)*len(all_zst)
    count = 1
    # fig,axes = plt.subplots(1, len(all_zst))
    for j in range(0, len(all_zst)):
        for i in range(0, len(all_b0)):

            path_multiple_mt =  f'/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/multiple_trees_treatment/RM_stdev/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/'

            isdir = os.path.isdir(path_multiple_mt)
            if isdir == False:
                os.makedirs(path_multiple_mt)

            path_multiple_zst = path_multiple_mt+f'zst_{all_zst[j]}/'

            isdir = os.path.isdir(path_multiple_zst)
            if isdir == False:
                os.makedirs(path_multiple_zst)

            path_multiple_b0 = path_multiple_zst + f'b0_{all_b0[i]}/'

            isdir = os.path.isdir(path_multiple_b0)
            if isdir == False:
                os.makedirs(path_multiple_b0)

            data_temp_L = np.empty((all_tree_number), dtype = np.ndarray)
            data_temp_M = np.empty((all_tree_number), dtype = np.ndarray)
            data_temp_U = np.empty((all_tree_number), dtype = np.ndarray)

            for tree_number in np.arange(1, all_tree_number + 1):

                path_to_plasma = path_to_mt + f'/logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}/tree_{tree_number}/'
                path_to_bf = path_to_plasma + f'bfields_data/mag_res_frac_1e{int(np.log10(mag_frac))}/b0_{all_b0[i]}/zst_{all_zst[j]}/'
                path_to_rm = path_to_bf + f'incr_factor_{incr_factor}/'

                z_eff = np.load(path_to_bf + 'z_eff.npy')
                np.save(path_multiple_b0+'z_eff.npy', z_eff)

                rm_L = np.load(path_to_rm + 'rm_L.npy', allow_pickle=True)
                rm_L_std = np.array([])

                rm_M = np.load(path_to_rm + 'rm_M.npy', allow_pickle=True)
                rm_M_std = np.array([])

                rm_U = np.load(path_to_rm + 'rm_U.npy', allow_pickle=True)
                rm_U_std = np.array([])

                for k in range(0, len(rm_L)):
                    rm_L_std = np.append(rm_L_std, np.std(rm_L[k][0]))
                    rm_M_std = np.append(rm_M_std, np.std(rm_M[k][0]))
                    rm_U_std = np.append(rm_U_std, np.std(rm_U[k][0]))

                data_temp_L[tree_number-1] = rm_L_std
                data_temp_M[tree_number-1] = rm_M_std
                data_temp_U[tree_number-1] = rm_U_std

            np.save(path_multiple_b0+'rm_L_std_all_trees.npy', data_temp_L)
            np.save(path_multiple_b0+'rm_M_std_all_trees.npy', data_temp_M)
            np.save(path_multiple_b0+'rm_U_std_all_trees.npy', data_temp_U)

            print(f'{count}/{total_op}')
            count+=1
def create_hdf5_file(dic_parameters):

    zmax = dic_parameters['zmax']
    zres = dic_parameters['zres']
    log_mclust = dic_parameters['log_mclust']
    log_mres = dic_parameters['log_mres']
    nb_real = dic_parameters['nb_real']
    resolution = dic_parameters['resolution']
    gamma = dic_parameters['gamma']
    fb = dic_parameters['fb']

    # Creating the HDF5 file
    # ------------------------------------------
    # path to merger trees data
    path_to_mergers_data = f'/home/yoan/Desktop/PHD/projects/project_I/MULTIPLE_TREES/mergers_data/mclust_1e{log_mclust}_zmax_{zmax}_zres_{zres}_mres_{log_mres}/'
    # path to store the hdf5 file
    path_to_store_data = '/home/yoan/Desktop/PHD/projects/project_II/data/processed_mts/'
    # name of the hdf5 file to store all data
    mt_name = f'logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}_data.hdf5'

    # test if the directory exists
    isdir = os.path.isdir(path_to_store_data)
    # if not, create it
    if isdir == False:
        os.mkdir(path_to_store_data)
    # Test if the hdf5 file already exists
    isfile = os.path.isfile(path_to_store_data+mt_name)
    # If not, create it
    if isfile == False:
        hfile = h5py.File(path_to_store_data+mt_name, 'w')
        hfile.close()

    #
    # # Group corresponding to zmax, zres, mclust and mres
    # hfile = h5py.File(path_to_store_data + mt_name, 'a')
    # hfile.require_group(f'logmclust_{log_mclust}_logmres_{log_mres}_zmax_{zmax}_zres_{zres}')
    # hfile.close()
