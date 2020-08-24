""" Sonar Probability Sensor Model (PSM) Code for ENMT482 assignment - Part A. """

###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################

def get_data():
    """ Get the calibration data. """
    # Load data
    filename = 'Part A/calibration.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    _, time, range_, _, _, _, _, _, sonar1, sonar2 = data.T

    # Convert into np arrays for ease.
    time = np.array(time)
    range_ = np.array(range_)
    sonar1 = np.array(sonar1)
    sonar2 = np.array(sonar2)


    return time, range_, sonar1, sonar2

###################################################################################################

def mod_Z_score(range_, sonar1, sonar2):
    """ Determine Iglewicz and Hoaglin's modified Z-score. """
    sonar1_error = range_ - sonar1
    sonar2_error = range_ - sonar2

    sorted_sonar1_error = np.sort(sonar1_error)
    sorted_sonar2_error = np.sort(sonar2_error)

    sorted_sonar1_error_abs = abs(sorted_sonar1_error)
    sorted_sonar2_error_abs = abs(sorted_sonar2_error)

    median_pos_sonar1 = int(len(sorted_sonar1_error) / 2) + 1
    median_pos_sonar2 = int(len(sorted_sonar2_error) / 2) + 1

    median_sonar1 = sorted_sonar1_error[median_pos_sonar1]
    median_sonar2 = sorted_sonar2_error[median_pos_sonar2]

    median_sonar1_abs = sorted_sonar1_error_abs[median_pos_sonar1]
    median_sonar2_abs = sorted_sonar2_error_abs[median_pos_sonar2]

    M_sonar1 = 0.6745 * (sonar1_error - median_sonar1) / median_sonar1_abs
    M_sonar2 = 0.6745 * (sonar2_error - median_sonar2) / median_sonar2_abs

    return M_sonar1, M_sonar2

def sonar_filter(time, range_, sonar1, sonar2):
    """ Implement a Iglewicz and Hoaglin's modified Z-score filter. """
    # Z-score.
    M_sonar1, M_sonar2 = mod_Z_score(range_, sonar1, sonar2)

    fil_range_sonar1 = []
    fil_time_sonar1 = []
    fil_values_sonar1 = []

    i = 0
    for M in M_sonar1:
        if abs(M) <= 3.5:
            fil_range_sonar1.append(range_[i])
            fil_time_sonar1.append(time[i])
            fil_values_sonar1.append(sonar1[i])
        else:
            None
        i += 1

    fil_range_sonar2 = []
    fil_time_sonar2 = []
    fil_values_sonar2 = []

    i = 0
    for M in M_sonar2:
        if abs(M) <= 3.5:
            fil_range_sonar2.append(range_[i])
            fil_time_sonar2.append(time[i])
            fil_values_sonar2.append(sonar2[i])
        else:
            None
        i += 1


    return fil_range_sonar1, fil_range_sonar2, fil_time_sonar1, fil_time_sonar2, fil_values_sonar1, fil_values_sonar2

###################################################################################################

def PDF(fil_range_sonar1, fil_range_sonar2, fil_values_sonar1, fil_values_sonar2):
    """ Work out the PDF's (mean and variance). """
    # Determine likelihoods.
    mean_sonar1 = sum(np.array(fil_values_sonar1) - np.array(fil_range_sonar1)) / len(fil_range_sonar1)
    mean_sonar2 = sum(np.array(fil_values_sonar2) - np.array(fil_range_sonar2)) / len(fil_range_sonar2)

    var_sonar1_array = []
    var_sonar2_array = []

    for val in fil_values_sonar1:
        var_sonar1_array.append((val - mean_sonar1) ** 2)

    for val in fil_values_sonar2:
        var_sonar2_array.append((val - mean_sonar2) ** 2)

    var_sonar1 = sum(var_sonar1_array) / len(var_sonar1_array)
    var_sonar2 = sum(var_sonar2_array) / len(var_sonar2_array)

    print("{}:{}\n{}:{}\n".format(mean_sonar1,var_sonar1,mean_sonar2,var_sonar2))

    f_v_sonar1 = []
    f_v_sonar2 = []

    for val in fil_values_sonar1:
        f_v_sonar1.append((1 / (2 * np.pi * var_sonar1)) * np.exp((-1/2) * ((val - mean_sonar1) ** 2) / (var_sonar1)))

    for val in fil_values_sonar2:
        f_v_sonar2.append((1 / (2 * np.pi * var_sonar2)) * np.exp((-1/2) * ((val - mean_sonar2) ** 2) / (var_sonar2)))

    return f_v_sonar1, f_v_sonar2

###################################################################################################

def plotting(fil_range_sonar1, fil_range_sonar2, fil_time_sonar1, fil_time_sonar2, fil_values_sonar1, fil_values_sonar2, f_v_sonar1, f_v_sonar2, time, range_, sonar1):
    """ Plot some results. """
    # Plot filtered data.
    plt.subplot(121)
    plt.plot(fil_range_sonar1, fil_values_sonar1,'.', alpha=0.2)

    plt.subplot(122)
    plt.plot(fil_range_sonar2, fil_values_sonar2,'.', alpha=0.2)


    # Plot true range and sonar measurements over time
    plt.figure(figsize=(12, 4))

    plt.subplot(141)
    plt.plot(time, range_)
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.title('True range')

    plt.subplot(142)
    plt.plot(fil_time_sonar1, fil_values_sonar1, '.', alpha=0.2)
    plt.plot(time, range_)
    plt.title('Sonar1')
    plt.xlabel('Time (s)')

    plt.subplot(143)
    plt.plot(fil_time_sonar2, fil_values_sonar2, '.', alpha=0.2)
    plt.plot(time, range_,)
    plt.title('Sonar2')
    plt.xlabel('Time (s)')


    # Plot PDFs against filtered data
    plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.plot(fil_values_sonar1, f_v_sonar1)
    plt.xlabel('Sonar 1 Filtered Data')
    plt.ylabel('PDF')
    plt.title('Sonar 1 PDF')

    plt.subplot(122)
    plt.plot(fil_values_sonar2, f_v_sonar2)
    plt.xlabel('Sonar 2 Filtered Data')
    plt.ylabel('PDF')
    plt.title('Sonar 2 PDF')


    plt.show()

###################################################################################################

def main():
    """ Main Function. """

    time, range_, sonar1, sonar2 = get_data()

    fil_range_sonar1, fil_range_sonar2, fil_time_sonar1, fil_time_sonar2, fil_values_sonar1, fil_values_sonar2 = sonar_filter(time, range_, sonar1, sonar2)

    f_v_sonar1, f_v_sonar2 = PDF(fil_range_sonar1, fil_range_sonar2, fil_values_sonar1, fil_values_sonar2)    

    plotting(fil_range_sonar1, fil_range_sonar2, fil_time_sonar1, fil_time_sonar2, fil_values_sonar1, fil_values_sonar2, f_v_sonar1, f_v_sonar2, time, range_, sonar1)

    for z in fil_range_sonar1:
        for x in fil_values_sonar1:
            

main()