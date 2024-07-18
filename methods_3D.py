from packages import *

def map_radial_distribution_onto_3d_matrix(injection_scale, rvir, val_distrib):

    ngrid = 2*int(rvir/injection_scale)
    grid_size = (ngrid, ngrid, ngrid)  # Size of the 3D grid
    center = (int(ngrid/2),int(ngrid/2),int(ngrid/2))       # Center of the distribution
    r_eff = np.linspace(0, int(ngrid/2), len(val_distrib))

    # Create an interpolation function
    interpolator = interp1d(r_eff, val_distrib, kind='linear', bounds_error=False, fill_value=0.0)
    # Create 3D coordinate grids
    x_coords = np.arange(grid_size[0])[:, np.newaxis, np.newaxis]
    y_coords = np.arange(grid_size[1])[np.newaxis, :, np.newaxis]
    z_coords = np.arange(grid_size[2])[np.newaxis, np.newaxis, :]
    # Calculate distances from the center for each voxel
    distances = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2 + (z_coords - center[2])**2)
    # Use the interpolation function to assign values to the 3D matrix
    matrix_3d = interpolator(distances)

    grid_size = rvir/int(ngrid/2)

    return grid_size, matrix_3d

def create_3d_equipartition_magnetic_field(injection_scale, r_range, rvir, equ_bfield):

    grid_size_equ_bf, matrix_3d_equ_bf = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, equ_bfield)
    beq_flat = matrix_3d_equ_bf.flatten()
    ngrid = matrix_3d_equ_bf.shape[0]

    a = -1
    b = 1

    random_numbers_x = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a
    random_numbers_y = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a
    random_numbers_z = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a

    indices_zero = np.where(matrix_3d_equ_bf.flatten()==0)[0]
    indices_nz = np.where(matrix_3d_equ_bf.flatten()!=0)[0]
    matrix_ones = np.ones((ngrid, ngrid, ngrid)).flatten()
    np.put(matrix_ones, indices_zero,0)
    matrix_ones = np.reshape(matrix_ones, (ngrid, ngrid, ngrid))

    result_x_temp = random_numbers_x * matrix_ones
    result_y_temp = random_numbers_y * matrix_ones
    result_z_temp = random_numbers_z * matrix_ones

    # CALCULATING THE GRADIENT OF THE A-VECTOR FIELD
    df_dy = np.gradient(result_z_temp, axis=2) - np.gradient(result_y_temp, axis=0)
    df_dz = np.gradient(result_x_temp, axis=0) - np.gradient(result_z_temp, axis=1)
    df_dx = np.gradient(result_y_temp, axis=1) - np.gradient(result_x_temp, axis=2)
    #
    df_dy = df_dy*matrix_ones
    df_dz = df_dz*matrix_ones
    df_dx = df_dx*matrix_ones

    result_x = df_dy.flatten()
    result_y = df_dz.flatten()
    result_z = df_dx.flatten()

    former_ampl = np.sqrt(result_x**2+result_y**2+result_z**2)

    temp_x = np.take(result_x, indices_nz)
    temp_y = np.take(result_y, indices_nz)
    temp_z = np.take(result_z, indices_nz)
    temp_beq = np.take(beq_flat, indices_nz)
    temp_ampl = np.take(former_ampl, indices_nz)

    norm_x = temp_x*(temp_beq/temp_ampl)
    norm_y = temp_y*(temp_beq/temp_ampl)
    norm_z = temp_z*(temp_beq/temp_ampl)

    np.put(result_x,  indices_nz, norm_x)
    result_x = np.reshape(result_x, (ngrid, ngrid, ngrid))

    np.put(result_y, indices_nz, norm_y)
    result_y = np.reshape(result_y, (ngrid, ngrid, ngrid))

    np.put(result_z, indices_nz, norm_z)
    result_z = np.reshape(result_z, (ngrid, ngrid, ngrid))

    ampl = np.sqrt(result_x**2+result_y**2+result_z**2)

    return result_x, result_y, result_z

def create_3d_magnetic_field_based_on_radial_distrib(injection_scale, r_range, rvir, equ_bfield, bfield, incr_fac):

    grid_size_equ_bf, matrix_3d_equ_bf = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, equ_bfield)
    grid_size_bf, matrix_3d_bf = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, bfield)


    # # ////////////creation of 3x 3d matrices that correspond to each component of the magnetic field ////////////////////
    ngrid = 2 * int(rvir / injection_scale)

    a = -1
    b = 1

    random_numbers_x = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a
    random_numbers_y = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a
    random_numbers_z = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a

    norms = np.sqrt(random_numbers_x ** 2 + random_numbers_y ** 2 + random_numbers_z ** 2)

    normalized_matrix_x = random_numbers_x / norms
    normalized_matrix_y = random_numbers_y / norms
    normalized_matrix_z = random_numbers_z / norms

    # normalized_matrix_x = random_numbers_x
    # normalized_matrix_y = random_numbers_y
    # normalized_matrix_z = random_numbers_z

    result_x_temp = normalized_matrix_x * matrix_3d_bf
    result_y_temp = normalized_matrix_y * matrix_3d_bf
    result_z_temp = normalized_matrix_z * matrix_3d_bf

    # CALCULATING THE GRADIENT OF THE A-VECTOR FIELD
    df_dy = np.gradient(result_z_temp, axis=0) - np.gradient(result_y_temp, axis=2)
    df_dz = np.gradient(result_x_temp, axis=1) - np.gradient(result_z_temp, axis=0)
    df_dx = np.gradient(result_y_temp, axis=2) - np.gradient(result_x_temp, axis=1)

    result_x = df_dy
    result_y = df_dz
    result_z = df_dx

    # ********** testing the indices that respect the condition of equipartition ***************s
    tol_equal = 0.1

    flat_beq = matrix_3d_equ_bf.flatten()
    flat_bf = matrix_3d_bf.flatten()

    ind = np.argwhere((flat_bf<=(1 - tol_equal)*flat_beq)|(flat_bf >= (1 + tol_equal)*flat_beq))
    corr_values = flat_bf[ind]

    if len(ind) > 0:
        # print('refinement')

        original_bequ_shape = matrix_3d_equ_bf.shape
        new_shape = tuple(np.array(original_bequ_shape) * incr_fac)

        expanded_matrix_beq = np.repeat(np.repeat(np.repeat(matrix_3d_equ_bf, incr_fac, axis=0), incr_fac, axis=1),incr_fac, axis=2)
        expanded_matrix_bf = np.repeat(np.repeat(np.repeat(matrix_3d_bf, incr_fac, axis=0), incr_fac, axis=1), incr_fac,axis=2)

        expanded_matrix_bx = np.repeat(np.repeat(np.repeat(result_x, incr_fac, axis=0), incr_fac, axis=1), incr_fac, axis=2)
        expanded_matrix_by = np.repeat(np.repeat(np.repeat(result_y, incr_fac, axis=0), incr_fac, axis=1), incr_fac, axis=2)
        expanded_matrix_bz = np.repeat(np.repeat(np.repeat(result_z, incr_fac, axis=0), incr_fac, axis=1), incr_fac, axis=2)

        flat_expanded_bf = expanded_matrix_bf.flatten()
        flat_expanded_bx = expanded_matrix_bx.flatten()
        flat_expanded_by = expanded_matrix_by.flatten()
        flat_expanded_bz = expanded_matrix_bz.flatten()

        ind_to_incr = np.where(np.isin(flat_expanded_bf, corr_values))[0]

        seg_rand_x = (b - a) * np.random.rand(len(ind_to_incr)) + a
        seg_rand_y = (b - a) * np.random.rand(len(ind_to_incr)) + a
        seg_rand_z = (b - a) * np.random.rand(len(ind_to_incr)) + a

        norm_seg = np.sqrt(seg_rand_x ** 2 + seg_rand_y ** 2 + seg_rand_z ** 2)

        # seg_rand_x = (seg_rand_x / norm_seg) * flat_expanded_bf[ind_to_incr]
        # seg_rand_y = (seg_rand_y / norm_seg) * flat_expanded_bf[ind_to_incr]
        # seg_rand_z = (seg_rand_z / norm_seg) * flat_expanded_bf[ind_to_incr]

        seg_rand_x = (seg_rand_x / norm_seg) * flat_expanded_bf[ind_to_incr]
        seg_rand_y = (seg_rand_y / norm_seg) * flat_expanded_bf[ind_to_incr]
        seg_rand_z = (seg_rand_z / norm_seg) * flat_expanded_bf[ind_to_incr]

        # seg_rand_x = (seg_rand_x / flat_expanded_bf[ind_to_incr]) * norm_seg
        # seg_rand_y = (seg_rand_y / flat_expanded_bf[ind_to_incr]) * norm_seg
        # seg_rand_z = (seg_rand_z / flat_expanded_bf[ind_to_incr]) * norm_seg
        #
        flat_expanded_bx[ind_to_incr] = seg_rand_x
        flat_expanded_by[ind_to_incr] = seg_rand_y
        flat_expanded_bz[ind_to_incr] = seg_rand_z

        # Reshape the modified flat array back to the expanded shape
        modified_expanded_bx = flat_expanded_bx.reshape(new_shape)
        modified_expanded_by = flat_expanded_by.reshape(new_shape)
        modified_expanded_bz = flat_expanded_bz.reshape(new_shape)

        grid_size = rvir/(int(modified_expanded_bx.shape[0]/2))

        return grid_size, modified_expanded_bx, modified_expanded_by, modified_expanded_bz

    else:
        # print('no refinement')
        grid_size = rvir/(int(result_x.shape[0]/2))
        return grid_size, result_x, result_y, result_z
def create_3d_magnetic_field_based_on_radial_distrib_array_of_incr_fac(injection_scale, r_range, rvir, equ_bfield, bfield, array_incr_fac):

    grid_size_equ_bf, matrix_3d_equ_bf = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, equ_bfield)
    grid_size_bf, matrix_3d_bf = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, bfield)

    all_matrix = np.empty((len(array_incr_fac)), dtype = dict)
    # # ////////////creation of 3x 3d matrices that correspond to each component of the magnetic field ////////////////////
    ngrid = 2 * int(rvir / injection_scale)

    a = -1
    b = 1

    random_numbers_x = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a
    random_numbers_y = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a
    random_numbers_z = (b - a) * np.random.rand(ngrid, ngrid, ngrid) + a

    norms = np.sqrt(random_numbers_x ** 2 + random_numbers_y ** 2 + random_numbers_z ** 2)

    normalized_matrix_x = random_numbers_x / norms
    normalized_matrix_y = random_numbers_y / norms
    normalized_matrix_z = random_numbers_z / norms

    result_x_temp = normalized_matrix_x * matrix_3d_bf
    result_y_temp = normalized_matrix_x * matrix_3d_bf
    result_z_temp = normalized_matrix_x * matrix_3d_bf

    # CALCULATING THE GRADIENT OF THE A-VECTOR FIELD
    df_dy = np.gradient(result_z_temp, axis=0) - np.gradient(result_y_temp, axis=2)
    df_dz = np.gradient(result_x_temp, axis=1) - np.gradient(result_z_temp, axis=0)
    df_dx = np.gradient(result_y_temp, axis=2) - np.gradient(result_x_temp, axis=1)

    result_x = df_dy
    result_y = df_dz
    result_z = df_dx

    # ********** testing the indices that respect the condition of equipartition ***************s
    tol_equal = 0.9

    flat_beq = matrix_3d_equ_bf.flatten()
    flat_bf = matrix_3d_bf.flatten()

    ind = np.argwhere((flat_bf<=(1 - tol_equal)*flat_beq)|(flat_bf >= (1 + tol_equal)*flat_beq))
    corr_values = flat_bf[ind]

    if len(ind) > 0:

        for i in range(0, len(array_incr_fac)):

            incr_fac = array_incr_fac[i]

            original_bequ_shape = matrix_3d_equ_bf.shape
            new_shape = tuple(np.array(original_bequ_shape) * incr_fac)

            expanded_matrix_beq = np.repeat(np.repeat(np.repeat(matrix_3d_equ_bf, incr_fac, axis=0), incr_fac, axis=1),incr_fac, axis=2)
            expanded_matrix_bf = np.repeat(np.repeat(np.repeat(matrix_3d_bf, incr_fac, axis=0), incr_fac, axis=1), incr_fac,axis=2)

            expanded_matrix_bx = np.repeat(np.repeat(np.repeat(result_x, incr_fac, axis=0), incr_fac, axis=1), incr_fac, axis=2)
            expanded_matrix_by = np.repeat(np.repeat(np.repeat(result_y, incr_fac, axis=0), incr_fac, axis=1), incr_fac, axis=2)
            expanded_matrix_bz = np.repeat(np.repeat(np.repeat(result_z, incr_fac, axis=0), incr_fac, axis=1), incr_fac, axis=2)

            flat_expanded_bf = expanded_matrix_bf.flatten()
            flat_expanded_bx = expanded_matrix_bx.flatten()
            flat_expanded_by = expanded_matrix_by.flatten()
            flat_expanded_bz = expanded_matrix_bz.flatten()

            ind_to_incr = np.where(np.isin(flat_expanded_bf, corr_values))[0]

            seg_rand_x = (b - a) * np.random.rand(len(ind_to_incr)) + a
            seg_rand_y = (b - a) * np.random.rand(len(ind_to_incr)) + a
            seg_rand_z = (b - a) * np.random.rand(len(ind_to_incr)) + a

            norm_seg = np.sqrt(seg_rand_x ** 2 + seg_rand_y ** 2 + seg_rand_z ** 2)

            seg_rand_x = (seg_rand_x / norm_seg) * flat_expanded_bf[ind_to_incr]
            seg_rand_y = (seg_rand_y / norm_seg) * flat_expanded_bf[ind_to_incr]
            seg_rand_z = (seg_rand_z / norm_seg) * flat_expanded_bf[ind_to_incr]
            #
            flat_expanded_bx[ind_to_incr] = seg_rand_x
            flat_expanded_by[ind_to_incr] = seg_rand_y
            flat_expanded_bz[ind_to_incr] = seg_rand_z

            # Reshape the modified flat array back to the expanded shape
            modified_expanded_bx = flat_expanded_bx.reshape(new_shape)
            modified_expanded_by = flat_expanded_by.reshape(new_shape)
            modified_expanded_bz = flat_expanded_bz.reshape(new_shape)

            grid_size = rvir/(int(modified_expanded_bx.shape[0]/2))

            all_matrix[i] = {'incr_fac': incr_fac, 'bx': modified_expanded_bx, 'by': modified_expanded_by, 'bz': modified_expanded_bz}

    else:

        grid_size = rvir/(int(result_x.shape[0]/2))
        all_matrix[i] = {'incr_fac': incr_fac, 'bx': result_x, 'by': result_y, 'bz': result_z}

    return all_matrix

def compute_rotation_measure_map(injection_scale, r_range, rvir, equ_bfield, bfield, incr_fac, particle_density_distrib, axis):
# def compute_rotation_measure_map(injection_scale, rvir, magfield_distrib, particle_density_distrib, axis):

    grid_size_bf, bf_x, bf_y, bf_z = create_3d_magnetic_field_based_on_radial_distrib(injection_scale, r_range, rvir, equ_bfield, bfield, incr_fac)
    grid_size_part, part_dens_3d = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, particle_density_distrib)

    # if part_dens_3d.shape != bf_x.shape:
    #     part_dens_3d = np.repeat(np.repeat(np.repeat(part_dens_3d, incr_fac, axis=0), incr_fac, axis=1),incr_fac, axis=2)

    bx, by, bz = create_3d_equipartition_magnetic_field(injection_scale, r_range, rvir, equ_bfield)
    if axis == 'x':
        temp_matrix = part_dens_3d*(bx*1e6)*(grid_size_bf*cm_to_pc)
        # temp_matrix = part_dens_3d*(bf_x*1e6)*(grid_size_bf*cm_to_pc)
        sum_temp = np.sum(temp_matrix, axis = 0)
        # return 0.81*sum_temp, sum_temp.shape, bf_x
        return 0.81*sum_temp, sum_temp.shape, bx

    elif axis == 'y':
        # temp_matrix = part_dens_3d*(bf_y*1e6)*(grid_size_bf*cm_to_pc)
        temp_matrix = part_dens_3d*(by*1e6)*(grid_size_bf*cm_to_pc)
        sum_temp = np.sum(temp_matrix, axis = 1)
        return 0.81*sum_temp, sum_temp.shape, by
        # return 0.81*sum_temp, sum_temp.shape, bf_y

    elif axis == 'z':
        temp_matrix = part_dens_3d*(bz*1e6)*(grid_size_bf*cm_to_pc)
        # temp_matrix = part_dens_3d*(bf_z*1e6)*(grid_size_bf*cm_to_pc)
        sum_temp = np.sum(temp_matrix, axis = 2)
        return 0.81*sum_temp, sum_temp.shape, bz
        # return 0.81*sum_temp, sum_temp.shape, bf_z
def compute_rotation_measure_map_all_axis(injection_scale, r_range, rvir, equ_bfield, bfield, incr_fac, particle_density_distrib):

    grid_size_bf, bf_x, bf_y, bf_z = create_3d_magnetic_field_based_on_radial_distrib(injection_scale, r_range, rvir, equ_bfield, bfield, incr_fac)
    grid_size_part, part_dens_3d = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, particle_density_distrib)

    if part_dens_3d.shape != bf_x.shape:
        part_dens_3d = np.repeat(np.repeat(np.repeat(part_dens_3d, incr_fac, axis=0), incr_fac, axis=1),incr_fac, axis=2)


    temp_matrix_x = part_dens_3d * (bf_x * 1e6) * (grid_size_bf * cm_to_pc)
    sum_temp_x = np.sum(temp_matrix_x, axis = 0)
    rmx = 0.81*sum_temp_x

    temp_matrix_y = part_dens_3d * (bf_y * 1e6) * (grid_size_bf * cm_to_pc)
    sum_temp_y = np.sum(temp_matrix_y, axis = 0)
    rmy = 0.81*sum_temp_y

    temp_matrix_z = part_dens_3d * (bf_z * 1e6) * (grid_size_bf * cm_to_pc)
    sum_temp_z = np.sum(temp_matrix_z, axis = 0)
    rmz = 0.81*sum_temp_z

    return rmx, rmy, rmz
def compute_rotation_measure_map_array_of_incr_fac(injection_scale, r_range, rvir, equ_bfield, bfield, array_incr_fac, particle_density_distrib, axis):

# def compute_rotation_measure_map(injection_scale, rvir, magfield_distrib, particle_density_distrib, axis):

    all_bf = create_3d_magnetic_field_based_on_radial_distrib_array_of_incr_fac(injection_scale, r_range, rvir, equ_bfield, bfield, array_incr_fac)
    all_rm = np.empty((len(all_bf)), dtype = dict)

    for i in range(0, len(all_bf)):

        bf_x = all_bf[i]['bx']
        bf_y = all_bf[i]['by']
        bf_z = all_bf[i]['bz']

        incr_fac = array_incr_fac[i]

        grid_size_part, part_dens_3d = map_radial_distribution_onto_3d_matrix(injection_scale, rvir, particle_density_distrib)

        if part_dens_3d.shape != bf_x.shape:
            part_dens_3d = np.repeat(np.repeat(np.repeat(part_dens_3d, incr_fac, axis=0), incr_fac, axis=1),incr_fac, axis=2)
            grid_size_bf = (2*rvir)/part_dens_3d.shape[0]
        else:
            grid_size_bf = (2*rvir)/part_dens_3d.shape[0]

        if axis == 'x':
            temp_matrix = part_dens_3d*(bf_x*1e6)*(grid_size_bf*cm_to_pc)
            sum_temp = np.sum(temp_matrix, axis = 0)
            # return 0.81*sum_temp, sum_temp.shape
            all_rm[i] = {'incr_fac': all_bf[i]['incr_fac'], 'rm': 0.81*sum_temp}

        elif axis == 'y':
            temp_matrix = part_dens_3d*(bf_y*1e6)*(grid_size_bf*cm_to_pc)
            sum_temp = np.sum(temp_matrix, axis = 0)
            # return 0.81*sum_temp, sum_temp.shape
            all_rm[i] = {'incr_fac': all_bf[i]['incr_fac'], 'rm': 0.81 * sum_temp}

        elif axis == 'z':
            temp_matrix = part_dens_3d*(bf_z*1e6)*(grid_size_bf*cm_to_pc)
            sum_temp = np.sum(temp_matrix, axis = 0)
            # return 0.81*sum_temp, sum_temp.shape
            all_rm[i] = {'incr_fac': all_bf[i]['incr_fac'], 'rm': 0.81 * sum_temp}

    return all_rm