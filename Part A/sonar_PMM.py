""" Determine the probability motion model (PMM) for the sonars. """

###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################

def get_data():
    """ Get the calibration data. """
    # Load data
    filename1 = 'Part A/training1.csv'
    data1 = np.loadtxt(filename1, delimiter=',', skiprows=1)
    filename2 = 'Part A/training2.csv'
    data2 = np.loadtxt(filename2, delimiter=',', skiprows=1)

    # Split into columns
    index_1, time_1, range_1, velocity_command_1, raw_ir1_1, raw_ir2_1, raw_ir3_1, raw_ir4_1, sonar1_1, sonar2_1 = data1.T
    index_2, time_2, range_2, velocity_command_2, raw_ir1_2, raw_ir2_2, raw_ir3_2, raw_ir4_2, sonar1_2, sonar2_2 = data2.T

    # Convert into np arrays for ease.
    time1 = np.array(time_1)
    range_1 = np.array(range_1)
    velocity_command_1 = np.array(velocity_command_1)
    sonar1_1 = np.array(sonar1_1)
    sonar2_1 = np.array(sonar2_1)
    time2 = np.array(time_2)
    range_2 = np.array(range_2)
    velocity_command_2 = np.array(velocity_command_2)
    sonar1_2 = np.array(sonar1_2)
    sonar2_2 = np.array(sonar2_2)
    


    return time1, range_1, velocity_command_1, sonar1_1, sonar2_1, time2, range_2, velocity_command_2, sonar1_2, sonar2_2
###################################################################################################

def c:
    """ Convert the data into velocities. v = d/t. """
    velocity_1 = []
    velocity_sonar1_1 = []
    velocity_sonar2_1 = []   

    for i in range(0, len(sonar2_1) - 1):
        
        velocity_sonar1_1.append((sonar1_1[i+1] - sonar1_1[i]) / (time1[i+1] - time1[i]))

        velocity_sonar2_1.append((sonar2_1[i+1] - sonar2_1[i]) / (time1[i+1] - time1[i]))

        velocity_1.append((range_1[i+1] - range_1[i]) / (time1[i+1] - time1[i]))

    velocity_2 = []
    velocity_sonar1_2 = []
    velocity_sonar2_2 = []

    for i in range(0, len(sonar2_1) - 1):
        
        velocity_sonar1_2.append((sonar1_2[i+1] - sonar1_2[i]) / (time2[i+1] - time2[i]))

        velocity_sonar2_2.append((sonar2_2[i+1] - sonar2_2[i]) / (time2[i+1] - time2[i]))

        velocity_2.append((range_2[i+1] - range_2[i]) / (time2[i+1] - time2[i]))


    return velocity_1, velocity_sonar1_1, velocity_sonar2_1, velocity_2, velocity_sonar1_2, velocity_sonar2_2

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

def sonar_velocity_filter(time1, range_1, velocity_command_1, velocity_1, velocity_sonar1_1, velocity_sonar2_1, time2, range_2, velocity_command_2, velocity_2, velocity_sonar1_2, velocity_sonar2_2):
    """ Implement a Iglewicz and Hoaglin's modified Z-score filter. """
    # Z-score.
    M_sonar1, M_sonar2 = mod_Z_score(velocity_1, velocity_sonar1_1, velocity_sonar2_1, velocity_2, velocity_sonar1_2, velocity_sonar2_2)

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

def plotting(time1, time2, velocity_command_1, velocity_command_2, velocity_sonar2_1, velocity_sonar1_1, velocity_1, sonar1_1, sonar2_1, range_1):
    """ Plot the outputs. """
    plt.figure()
    plt.subplot(141)
    plt.plot(time1[1:], velocity_1, '-', alpha=0.8)
    plt.plot(time1, velocity_command_1, '-', alpha=0.8, color='y')
    
    plt.subplot(142)
    #plt.plot(time1[1:], velocity_sonar1_1, '.', alpha=0.8, color='b')
    plt.plot(time1[1:], velocity_sonar2_1, '.', alpha=0.8, color='y')

    plt.subplot(143)
    plt.plot(time2, velocity_command_2, '-', alpha=0.8)

    plt.subplot(144)
    plt.plot(range_1, sonar1_1, '.', alpha=0.8, color='b')
    plt.plot(range_1, sonar2_1, '.', alpha=0.8, color='y')

    plt.show()

###################################################################################################

def main():
    """ Main Function. """
    # Get the data from csv and return required dataset vectors.
    time1, range_1, velocity_command_1, sonar1_1, sonar2_1, time2, range_2, velocity_command_2, sonar1_2, sonar2_2 = get_data()

    velocity_1, velocity_sonar1_1, velocity_sonar2_1, velocity_2, velocity_sonar1_2, velocity_sonar2_2 = time1, range_1, velocity_command_1, sonar1_1, sonar2_1, time2, range_2, velocity_command_2, sonar1_2, sonar2_2

    sonar_velocity_filter(time1, range_1, velocity_command_1, velocity_1, velocity_sonar1_1, velocity_sonar2_1, time2, range_2, velocity_command_2, velocity_2, velocity_sonar1_2, velocity_sonar2_2)

    plotting(time1, time2, velocity_command_1, velocity_command_2,velocity_sonar2_1, velocity_sonar1_1, velocity_1, sonar1_1, sonar2_1, range_1)
    

main()