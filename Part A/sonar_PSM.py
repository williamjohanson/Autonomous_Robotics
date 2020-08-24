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
        filtered_sonar1[(range_[i], time[i])] = sonar1[i]
    else:
        None
    i += 1

i = 0
for M in M_sonar2:
    if abs(M) <= 3.5:
        filtered_sonar2[(range_[i], time[i])] = sonar2[i]
    else:
        None
    i += 1

filtered_range_sonar1 = []
filtered_time_sonar1 = []
filtered_values_sonar1 = []

for (range_, time), values_sonar1 in filtered_sonar1.items():
    filtered_range_sonar1.append(range_)
    filtered_time_sonar1.append(time)
    filtered_values_sonar1.append(values_sonar1)

filtered_range_sonar2 = []
filtered_time_sonar2 = []
filtered_values_sonar2 = []

for (range_, time), values_sonar2 in filtered_sonar2.items():
    filtered_range_sonar2.append(range_)
    filtered_time_sonar2.append(time)
    filtered_values_sonar2.append(values_sonar2)

###################################################################################################

""" Work out the PDF's (mean and variance). """

# Determine likelihoods.
mean_sonar1 = sum(np.array(filtered_values_sonar1) - np.array(filtered_range_sonar1)) / len(filtered_range_sonar1)
mean_sonar2 = sum(np.array(filtered_values_sonar2) - np.array(filtered_range_sonar2)) / len(filtered_range_sonar2)

var_sonar1_array = []
var_sonar2_array = []

for val in filtered_values_sonar1:
    var_sonar1_array.append((val - mean_sonar1) ** 2)

for val in filtered_values_sonar2:
    var_sonar2_array.append((val - mean_sonar2) ** 2)

var_sonar1 = sum(var_sonar1_array) / len(var_sonar1_array)
var_sonar2 = sum(var_sonar2_array) / len(var_sonar2_array)

print("{}:{}\n{}:{}\n".format(mean_sonar1,var_sonar1,mean_sonar2,var_sonar2))

f_v_sonar1 = []
f_v_sonar2 = []

for val in filtered_values_sonar1:
    f_v_sonar1.append((1 / (2 * np.pi * var_sonar1)) * np.exp((-1/2) * ((val - mean_sonar1) ** 2) / (var_sonar1)))

for val in filtered_values_sonar2:
    f_v_sonar2.append((1 / (2 * np.pi * var_sonar2)) * np.exp((-1/2) * ((val - mean_sonar2) ** 2) / (var_sonar2)))

###################################################################################################

""" Plotting. """

# Plot filtered data.
plt.subplot(121)
plt.plot(filtered_range_sonar1, filtered_values_sonar1,'.', alpha=0.2)

plt.subplot(122)
plt.plot(filtered_range_sonar2, filtered_values_sonar2,'.', alpha=0.2)


# Plot true range and sonar measurements over time
plt.figure(figsize=(12, 4))

plt.subplot(141)
plt.plot(time, range_)
plt.xlabel('Time (s)')
plt.ylabel('Range (m)')
plt.title('True range')

plt.subplot(142)
plt.plot(filtered_time_sonar1, filtered_values_sonar1, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar1')
plt.xlabel('Time (s)')

plt.subplot(143)
plt.plot(filtered_time_sonar2, filtered_values_sonar2, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar2')
plt.xlabel('Time (s)')


# Plot PDFs against filtered data
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.plot(filtered_values_sonar1, f_v_sonar1)
plt.xlabel('Sonar 1 Filtered Data')
plt.ylabel('PDF')
plt.title('Sonar 1 PDF')

plt.subplot(122)
plt.plot(filtered_values_sonar2, f_v_sonar2)
plt.xlabel('Sonar 2 Filtered Data')
plt.ylabel('PDF')
plt.title('Sonar 2 PDF')



plt.show()