""" Sonar Probability Motion Model (PMM) Code for ENMT482 assignment - Part A. 

Xn = Xn-1 + un * deltaT + Wn"""

###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt

###################################################################################################

def get_data():
    """ Get the calibration data. """
    # Load data
    filename = 'Part A/training1.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    _, time, range_, velocity_command, _, _, _, _, sonar1, sonar2 = data.T

    # Convert into np arrays for ease.
    time = np.array(time)
    range_ = np.array(range_)
    velocity_command = np.array(velocity_command)
    sonar1 = np.array(sonar1)
    sonar2 = np.array(sonar2)


    return time, range_, velocity_command, sonar1, sonar2

###################################################################################################

def mod_Z_score(range_, disp_command, sonar1, sonar2):
    """ Determine Iglewicz and Hoaglin's modified Z-score. """
    sonar1_error = range_ - (sonar1 + disp_command)
    sonar2_error = range_ - (sonar2 + disp_command)

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

def sonar_filter(time, range_, disp_command, sonar1, sonar2):
    """ Implement a Iglewicz and Hoaglin's modified Z-score filter. """
    # Z-score.
    M_sonar1, M_sonar2 = mod_Z_score(range_, disp_command, sonar1, sonar2)

    fil_range_sonar1 = []
    fil_time_sonar1 = []  
    fil_disp_command_sonar1 = []
    fil_sonar1 = []
    

    i = 0
    for M in M_sonar1:
        if abs(M) <= 3.5:
            fil_range_sonar1.append(range_[i])
            fil_time_sonar1.append(time[i])
            fil_sonar1.append(sonar1[i])
            fil_disp_command_sonar1.append(disp_command[i])
        else:
            None
        i += 1

    fil_range_sonar2 = []
    fil_time_sonar2 = []
    fil_disp_command_sonar2 = []
    fil_sonar2 = []

    i = 0
    for M in M_sonar2:
        if abs(M) <= 3.5:
            fil_range_sonar2.append(range_[i])
            fil_time_sonar2.append(time[i])
            fil_sonar2.append(sonar2[i])
            fil_disp_command_sonar2.append(disp_command[i])
        else:
            None
        i += 1


    return fil_range_sonar1, fil_range_sonar2, fil_time_sonar1, fil_time_sonar2, fil_sonar1, fil_sonar2, fil_disp_command_sonar1, fil_disp_command_sonar2

###################################################################################################

def vel_to_disp(time, velocity_command, sonar1):
    """ Convert the velocity command to displacement. """
    disp_command = [0]

    for i in range(0, len(sonar1)):
        try:
            disp_command.append(velocity_command[i+1] * (time[i+1] - time[i]))
        except IndexError:
            None


    return disp_command

###################################################################################################

def linear_least_squares(fil_range_sonar1, fil_range_sonar2, fil_sonar1, fil_sonar2, fil_disp_command_sonar1, fil_disp_command_sonar2):
    """ Determine the variables A and B from Xn=AXn-1+Bun. """

    Y_1 = fil_range_sonar1
    Y_2 = fil_range_sonar2

    A_1 = np.transpose(np.array([fil_sonar1, fil_disp_command_sonar1]))
    A_2 = np.transpose(np.array([fil_sonar2, fil_disp_command_sonar2]))

    trans_A_1 = np.transpose(A_1)
    trans_A_2 = np.transpose(A_2)

    theta_1 = np.dot(np.dot(np.linalg.inv(np.dot(trans_A_1, A_1)), trans_A_1), Y_1)
    theta_2 = np.dot(np.dot(np.linalg.inv(np.dot(trans_A_2, A_2)), trans_A_2), Y_2)


    return theta_1, theta_2

###################################################################################################

def error_PDF(x, x2, x_est, x2_est):
    """ Work out the PDF's (mean and variance). """
    # Determine error PDF.
    error_sonar1 = np.array(x_est) - np.array(x)
    error_sonar2 = np.array(x2_est) - np.array(x2)

    mean_sonar1 = sum(error_sonar1) / len(error_sonar1)
    mean_sonar2 = sum(error_sonar2) / len(error_sonar2)

    var_sonar1_array = []
    var_sonar2_array = []

    for val in error_sonar1:
        var_sonar1_array.append((val - mean_sonar1) ** 2)

    for val in error_sonar2:
        var_sonar2_array.append((val - mean_sonar2) ** 2)

    var_sonar1 = sum(var_sonar1_array) / len(var_sonar1_array)
    var_sonar2 = sum(var_sonar2_array) / len(var_sonar2_array)

    print("{}:{}\n{}:{}\n".format(mean_sonar1,var_sonar1,mean_sonar2,var_sonar2))

    f_v_sonar1_plot = []
    f_v_sonar2_plot = []
    f_v_sonar1 = []
    f_v_sonar2 = []

    x_range = np.linspace(-5,5,1000)

    for val in x_range:
        f_v_sonar1_plot.append((1 / (2 * np.pi * var_sonar1)) * np.exp((-1/2) * ((val - mean_sonar1) ** 2) / (var_sonar1)))

    for val in x_range:
        f_v_sonar2_plot.append((1 / np.sqrt(2 * np.pi * var_sonar2)) * np.exp((-1/2) * ((val - mean_sonar2) ** 2) / (var_sonar2)))

    for val in x_est:
        f_v_sonar1.append((1 / (2 * np.pi * var_sonar1)) * np.exp((-1/2) * ((val - mean_sonar1) ** 2) / (var_sonar1)))

    for val in x2_est:
        f_v_sonar2.append((1 / np.sqrt(2 * np.pi * var_sonar2)) * np.exp((-1/2) * ((val - mean_sonar2) ** 2) / (var_sonar2)))


    return f_v_sonar1_plot, f_v_sonar2_plot, f_v_sonar1, f_v_sonar2, x_range

###################################################################################################

def robot_velocity(range_, time):
    velocity_array = []
    t0 = time[1]
    r0 = range_[1]
    N = len(time)
    for i in range(N):
        v = (range_[i] - r0) / (time[i] - t0)
        velocity_array.append(v)
        r0 = range_[i]
        t0 = time[i]
    
    return velocity_array

def approximates(fil_sonar1, fil_sonar2, fil_disp_command_sonar1, fil_disp_command_sonar2, theta_1, theta_2):
    """ Approximate data and determined data for plotting. """
    #sonar2_min = 0.3 # Min the sonar 2 can see.
    x = []
    x_est = []
    x.append(0)
    x_est.append(0)

    i = 0
    for val in fil_sonar1:
        #if range_[i] > sonar2_min:
        try:
            x_new = val + fil_disp_command_sonar1[i+1] 
            x_est_new = theta_1[0] * val + theta_1[1] * fil_disp_command_sonar1[i+1]
            x.append(x_new)
            x_est.append(x_est_new)
            i += 1
        except IndexError:
            None

    x2 = []
    x2_est = []
    x2.append(0)
    x2_est.append(0)
    j = 0
    for val in fil_sonar2:
        #if range_[i] > sonar2_min:
        try:
            x2_new = val + fil_disp_command_sonar2[j+1] 
            x2_est_new = theta_2[0] * val + theta_2[1] * fil_disp_command_sonar2[j+1]
            x2.append(x2_new)
            x2_est.append(x2_est_new)
            j += 1
        except IndexError:
            None


    return x, x2, x_est, x2_est

###################################################################################################

def plotting(time, fil_time_sonar1, fil_time_sonar2, range_, x, x2, x_est, x2_est,f_v_sonar1_plot, f_v_sonar2_plot, f_v_sonar1, f_v_sonar2, x_range, disp_command, velocity_command, robot_vel): 
    """ Plot some results. """
    plt.figure()
    plt.subplot(141)
    plt.plot(np.array(time), np.array(range_), '.', alpha=0.2, color='b')
    plt.plot(np.array(fil_time_sonar1), np.array(x), '.', alpha=0.2, color='y')
    plt.plot(np.array(fil_time_sonar1), np.array(x_est), '.', alpha=0.2, color='g')

    plt.subplot(142)
    plt.plot(np.array(time), np.array(range_), '.', alpha=0.2, color='b')
    plt.plot(np.array(fil_time_sonar2), np.array(x2), '.', alpha=0.2, color='y')
    plt.plot(np.array(fil_time_sonar2), np.array(x2_est), '.', alpha=0.2, color='g')

    plt.subplot(143)
    plt.plot(x_range, f_v_sonar1_plot)

    plt.subplot(144)
    plt.plot(x_range, f_v_sonar2_plot)

    plt.figure()
    plt.plot( time, robot_vel)
    plt.plot( time, velocity_command)
    plt.show()

###################################################################################################

def main():
    """ Main Function. """

    time, range_, velocity_command, sonar1, sonar2 = get_data()

    disp_command = vel_to_disp(time, velocity_command, sonar1)

    fil_range_sonar1, fil_range_sonar2, fil_time_sonar1, fil_time_sonar2, fil_sonar1, fil_sonar2, fil_disp_command_sonar1, fil_disp_command_sonar2 = sonar_filter(time, range_, disp_command, sonar1, sonar2)
    
    theta_1, theta_2 = linear_least_squares(fil_range_sonar1, fil_range_sonar2, fil_sonar1, fil_sonar2, fil_disp_command_sonar1, fil_disp_command_sonar2)

    print(theta_1, theta_2)

    x, x2, x_est, x2_est = approximates(fil_sonar1, fil_sonar2, fil_disp_command_sonar1, fil_disp_command_sonar2, theta_1, theta_2)      

    f_v_sonar1_plot, f_v_sonar2_plot, f_v_sonar1, f_v_sonar2, x_range = error_PDF(x, x2, x_est, x2_est)

    robot_vel = robot_velocity(range_, time)

    plotting(time, fil_time_sonar1, fil_time_sonar2, range_, x, x2, x_est, x2_est, f_v_sonar1_plot, f_v_sonar2_plot, f_v_sonar1, f_v_sonar2, x_range, disp_command, velocity_command, robot_vel)

    



 

######################################## Run the main func.########################################

###################################################################################################

main() ############################################################################################

###################################################################################################

###################################################################################################