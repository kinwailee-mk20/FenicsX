from utility import *
import os

# convert csv to numpy with noise - 1500 data
ins_x_clean = []
cond_x_clean = []
cond_x = []
cond_y = []
ins_x = []
ins_y = []

snr_ins = []
snr_cond = []

scale = 0
directory = '1.1_new/insulating/x'
for i in range(len(os.listdir(directory))):
    filename = str(i) + '_single_insulator_sphere_x.csv'
    csv_file = os.path.join(directory, filename)
    df = pd.read_csv(csv_file, header=None)
    data = df.values
    ins_x_clean.append(data.flatten())
ins_x_clean = np.array(ins_x_clean)
noise_only_array, ins_x_noise = add_noise(ins_x_clean, scale)


directory = '1.1_new/conducting/x'
for i in range(len(os.listdir(directory))):
    filename = str(i) + '_single_conductor_sphere_x.csv'
    csv_file = os.path.join(directory, filename)
    df = pd.read_csv(csv_file, header=None)
    data = df.values
    cond_x_clean.append(data.flatten())
cond_x_clean = np.array(cond_x_clean)
noise_only_array, cond_x_noise = add_noise(cond_x_clean, scale)


snr_ins = np.mean(ins_x_clean**2) / np.var(noise_only_array)
snr_cond = np.mean(cond_x_clean**2) / np.var(noise_only_array)
snr_mean = (snr_ins + snr_cond) / 2
snr_db = 10*np.log10(snr_mean)
print('snr_mean: ', snr_db)


directory = '1.1_new/insulating/y'
for i in range(len(os.listdir(directory))):
    filename = str(i) + '_single_insulator_sphere_y.csv'
    csv_file = os.path.join(directory, filename)
    df = pd.read_csv(csv_file, header=None)
    data = df.values
    ins_y.append(data.flatten())
ins_y = np.array(ins_y)


directory = '1.1_new/conducting/y'
for i in range(len(os.listdir(directory))):
    filename = str(i) + '_single_conductor_sphere_y.csv'
    csv_file = os.path.join(directory, filename)
    df = pd.read_csv(csv_file, header=None)
    data = df.values
    cond_y.append(data.flatten())
cond_y = np.array(cond_y)


noise_x, noise_y = get_noise(500, scale)
noise_x = noise_x.reshape((500, 128))


x = np.vstack((noise_x, cond_x_noise, ins_x_noise))
y = np.vstack((noise_y, cond_y, ins_y))
name = '1.2_noise_{}'.format(scale)
np.save(name + 'x.npy', x)
np.save(name + 'y.npy', y)


