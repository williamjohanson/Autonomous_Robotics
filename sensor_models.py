import numpy as np
from matplotlib.pyplot import subplots, show

# Load data
filename = 'Part A/calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Define arrays.
error_sonar1 = []
error_sonar2 = []
error_raw_ir3 = []
i_array = []

# Split into columns
index, time, distance, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, \
    sonar1, sonar2 = data.T

error_sonar1 = sonar1 - distance
error_sonar2 = sonar2 - distance
error_raw_ir3 = raw_ir3 - distance

for i in range(1, len(error_sonar1) + 1):
    i_array.append(i)

error_sonar1.sort()
error_sonar2.sort()
error_raw_ir3.sort()

fig, axes = subplots(3, 3)
fig.suptitle('Calibration data')

axes[0, 0].plot(distance, raw_ir1, '.', alpha=0.2)
axes[0, 0].set_title('IR1')

axes[0, 1].plot(distance, raw_ir2, '.', alpha=0.2)
axes[0, 1].set_title('IR2')

axes[0, 2].plot(distance, raw_ir3, '.', alpha=0.2)
axes[0, 2].set_title('IR3')

axes[1, 0].plot(distance, raw_ir4, '.', alpha=0.2)
axes[1, 0].set_title('IR4')

axes[1, 1].plot(distance, sonar1, '.', alpha=0.2)
axes[1, 1].set_title('Sonar1')

axes[1, 2].plot(distance, sonar2, '.', alpha=0.2)
axes[1, 2].set_title('Sonar2')

axes[2, 0].plot(i_array, error_sonar1, '.', alpha=0.2)
axes[2, 0].set_title('Sonar1 Error')

axes[2, 1].plot(i_array, error_sonar2, '.', alpha=0.2)
axes[2, 1].set_title('Sonar2 Error')

axes[2, 2].plot(i_array, error_raw_ir3, '.', alpha=0.2)
axes[2, 2].set_title('Raw IR3 Error')


show()






