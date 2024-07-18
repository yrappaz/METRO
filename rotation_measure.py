from packages import *
from external_methods import *
class RM_MAP:

    def __init__(self, data_file_B_path, bfield_params_path, data_ne, size_box_pc):

        # Get information from the parameters.txt file
        params_file = open(bfield_params_path, 'r')
        lines = params_file.readlines()

        for l in lines:
            temp = l.split(' ')
            label = temp[0]

            if label == 'ndim':
                self.ndim = int(temp[1])
            elif label == 'lx':
                self.lx = float(temp[1])
            elif label == 'ly':
                self.ly = float(temp[1])
            elif label == 'lz':
                self.lz = float(temp[1])
            elif label == 'kmin':
                self.kmin = float(temp[1])
            elif label == 'kmax':
                self.kmax = float(temp[1])
            elif label == 'spect_form':
                self.spect_form = int(temp[1])
            elif label == 'power_law_exp':
                self.power_law_exp = float(temp[1])
            elif label == 'angles_exp':
                self.angles_exp = float(temp[1])
            elif label == 'sol_weight':
                self.sol_weight = float(temp[1])
            elif label == 'ngrid':
                self.ngrid = int(temp[1])

        # Extraction of the magnetic field components from the data.txt file
        raw_data_B = np.loadtxt(data_file_B_path)

        bx = np.empty((self.ngrid, self.ngrid, self.ngrid), dtype=float)
        by = np.empty((self.ngrid, self.ngrid, self.ngrid), dtype=float)
        bz = np.empty((self.ngrid, self.ngrid, self.ngrid), dtype=float)
        #
        for i in range(0, len(raw_data_B)):
            cx, cy, cz = int(raw_data_B[i][0]), int(raw_data_B[i][1]), int(raw_data_B[i][2])

            bx[cx, cy, cz] = raw_data_B[i][3]
            by[cx, cy, cz] = raw_data_B[i][4]
            bz[cx, cy, cz] = raw_data_B[i][5]

        self.bx = bx
        self.by = by
        self.bz = bz
        self.b_ampl = np.sqrt(bx**2+by**2+bz**2)
        self.ne = data_ne

        # Computation of all rotation measure maps
        rm_map_x = np.zeros((self.ngrid, self.ngrid))
        rm_map_y = np.zeros((self.ngrid, self.ngrid))
        rm_map_z = np.zeros((self.ngrid, self.ngrid))

        grid_size_pc = size_box_pc/self.ngrid

        for i in range(0, self.ngrid):

            rm_map_x += 0.811 * data_ne[i,:,:] * (bx[i,:,:] * 1e6) * grid_size_pc
            rm_map_y += 0.811 * data_ne[:,i,:] * (by[:,i,:] * 1e6) * grid_size_pc
            rm_map_z += 0.811 * data_ne[:,:,i] * (bz[:,:,i] * 1e6) * grid_size_pc

        self.rm_map_x = rm_map_x
        self.rm_map_y = rm_map_y
        self.rm_map_z = rm_map_z

    def display_bfield_parameters(self):
        print("********************************************")
        print("Parameters of the generated magnetic field")
        print("********************************************")

        print("Number of dimension: ndim = "+str(self.ndim))
        print("Number of grid cells : ngrid = "+str(self.ngrid))
        print("x-axis size (2 pi / L): lx = "+str(self.lx))
        print("y-axis size (2 pi / L): ly = "+str(self.ly))
        print("z-axis size (2 pi / L): lz = "+str(self.lz))
        print("Min. wavenumber of turbulent modes: kmin = " + str(self.kmin))
        print("Max. wavenumber of turbulent modes: kmax = " + str(self.kmax))
        print("Power law exponent of the magnetic field's power spectrum : power_law_exp = " + str(self.power_law_exp))
        print("number of modes (angles) in k-shell surface increases as k^angles_exp : angles_exp = " + str(self.angles_exp))

    def calculate_statistics(self):
        # Average of each RM map
        avg_rm_x = np.mean(self.rm_map_x)
        avg_rm_y = np.mean(self.rm_map_y)
        avg_rm_z = np.mean(self.rm_map_z)

        # rms of each RM map
        rms_rm_x = calculate_rms(self.rm_map_x)
        rms_rm_y = calculate_rms(self.rm_map_y)
        rms_rm_z = calculate_rms(self.rm_map_z)

        # standard deviation of each RM map
        std_rm_x = np.std(self.rm_map_x)
        std_rm_y = np.std(self.rm_map_y)
        std_rm_z = np.std(self.rm_map_z)

        data = {
            'avg_rm_x': avg_rm_x,
            'avg_rm_y': avg_rm_y,
            'avg_rm_z': avg_rm_z,
            'rms_rm_x': rms_rm_x,
            'rms_rm_y': rms_rm_y,
            'rms_rm_z': rms_rm_z,
            'std_rm_x': std_rm_x,
            'std_rm_y': std_rm_y,
            'std_rm_z': std_rm_z
        }

        return data

    def plot_all_axis_rm(self):

        axis_labels = np.array(['x', 'y', 'z'])

        fig,axes = plt.subplots(2,3)

        axes[0,0].imshow(self.rm_map_x, cmap = 'coolwarm')
        axes[0,1].imshow(self.rm_map_y, cmap = 'coolwarm')
        axes[0,2].imshow(self.rm_map_z, cmap = 'coolwarm')

        axes[1,0].hist(self.rm_map_x.flatten(), bins = 100, histtype = 'step')
        axes[1,1].hist(self.rm_map_y.flatten(), bins = 100, histtype = 'step')
        axes[1,2].hist(self.rm_map_z.flatten(), bins = 100, histtype = 'step')









