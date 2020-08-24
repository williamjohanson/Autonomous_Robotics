"""Example code for ENMT482 assignment."""

###################################################################################################

""" Code setup. """
# Imports
import numpy as np
import matplotlib.pyplot as plt

# Load data
filename = 'Part A/calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# Convert into np arrays for ease.
range_ = np.array(range_)
raw_ir3 = np.array(raw_ir3)
sonar1 = np.array(sonar1)
sonar2 = np.array(sonar2)

###################################################################################################

""" Filtering Code for the sonars. """

# Error of the sonar measurements.
sonar1_error = range_ - sonar1
sonar2_error = range_ - sonar2

sorted_sonar1_error = np.sort(sonar1_error)
sorted_sonar2_error = np.sort(sonar2_error)

median_pos_sonar1 = int(len(sorted_sonar1_error) / 2) + 1
median_pos_sonar2 = int(len(sorted_sonar2_error) / 2) + 1

median_sonar1 = sorted_sonar1_error[median_pos_sonar1]
median_sonar2 = sorted_sonar2_error[median_pos_sonar2]

sorted_sonar1_error_abs = abs(sorted_sonar1_error)
sorted_sonar2_error_abs = abs(sorted_sonar2_error)

median_sonar1_abs = sorted_sonar1_error_abs[median_pos_sonar1]
median_sonar2_abs = sorted_sonar2_error_abs[median_pos_sonar2]

M_sonar1 = 0.6745 * (sonar1_error - median_sonar1) / median_sonar1_abs
M_sonar2 = 0.6745 * (sonar2_error - median_sonar2) / median_sonar2_abs

filtered_sonar1 = dict()
filtered_sonar2 = dict()

i = 0
for M in M_sonar1:
    if abs(M) <= 3.5:
        filtered_sonar1[range_[i]] = sonar1[i]
    else:
        None
    i += 1

i = 0
for M in M_sonar2:
    if abs(M) <= 3.5:
        filtered_sonar2[range_[i]] = sonar2[i]
    else:
        None
    i += 1


###################################################################################################

# Determine likelihoods.
mean_raw_ir3 = sum(raw_ir3 - range_) / len(raw_ir3)
mean_sonar1 = sum(filtered_sonar1.values() - filtered_sonar1.keys()) / len(filtered_sonar1.values())
mean_sonar2 = sum(filtered_sonar2.values() - filtered_sonar2.keys()) / len(filtered_sonar2.values())

var_raw_ir3_array = []
var_sonar1_array = []
var_sonar2_array = []

for i in range(0, len(range_)):
    var_raw_ir3_array.append((raw_ir3[i] - mean_raw_ir3) ** 2)
    var_sonar1_array.append((filtered_sonar1.values()[i] - mean_sonar1) ** 2) 
    var_sonar2_array.append((filtered_sonar1.values()[i] - mean_sonar2) ** 2)

var_raw_ir3 = sum(var_raw_ir3_array) / len(var_raw_ir3_array)
var_sonar1 = sum(var_sonar1_array) / len(var_sonar1_array)
var_sonar2 = sum(var_sonar2_array) / len(var_sonar2_array)

print("{}:{}\n{}:{}\n{}:{}\n".format(mean_raw_ir3,var_raw_ir3,mean_sonar1,var_sonar1,mean_sonar2,var_sonar2))

f_v_raw_ir3 = (1 / (2 * np.pi * var_raw_ir3)) * np.exp((-1/2) * ((raw_ir3 - mean_raw_ir3) ** 2) / (var_raw_ir3))
f_v_sonar1 = (1 / (2 * np.pi * var_sonar1)) * np.exp((-1/2) * ((sonar1 - mean_sonar1) ** 2) / (var_sonar1))
f_v_sonar2 = (1 / (2 * np.pi * var_sonar2)) * np.exp((-1/2) * ((sonar2 - mean_sonar2) ** 2) / (var_sonar2))

# Plot filtered data.
plt.subplot(121)
plt.plot(filtered_sonar1.keys(), filtered_sonar1.values(),'.', alpha=0.2)

plt.subplot(122)
plt.plot(filtered_sonar2.keys(), filtered_sonar2.values(),'.', alpha=0.2)


# Plot true range and sonar measurements over time
plt.figure(figsize=(12, 4))

plt.subplot(141)
plt.plot(time, range_)
plt.xlabel('Time (s)')
plt.ylabel('Range (m)')
plt.title('True range')

plt.subplot(142)
plt.plot(time, sonar1, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar1')
plt.xlabel('Time (s)')

plt.subplot(143)
plt.plot(time, sonar2, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar2')
plt.xlabel('Time (s)')

plt.subplot(144)
plt.plot(range_, raw_ir3, '.', alpha=0.5)
plt.title('IR3')
plt.xlabel('Range (m)')
plt.ylabel('Measurement (V)')
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(raw_ir3, f_v_raw_ir3)
plt.xlabel('IR3 Raw Data')
plt.ylabel('PDF')
plt.title('IR3 PDF')


plt.subplot(132)
plt.plot(sonar1, f_v_sonar1)
plt.xlabel('Sonar 1 Raw Data')
plt.ylabel('PDF')
plt.title('Sonar 1 PDF')


plt.subplot(133)
plt.plot(sonar2, f_v_sonar2)
plt.xlabel('Sonar 2 Raw Data')
plt.ylabel('PDF')
plt.title('Sonar 2 PDF')



plt.show()