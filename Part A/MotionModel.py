###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, zeros, linspace

###################################################################################################

def get_data():
    """ Get the calibration data. """
    # Load data
    filename = 'Part A/training1.csv'
    data = np.loadtxt(filename, delimiter=',', skiprows=1)

    # Split into columns
    _, time, range_, velocity_command, _, _, _, _, _, _ = data.T

    # Convert into np arrays for ease.
    time = np.array(time)
    range_ = np.array(range_)
    velocity_command = np.array(velocity_command)


    return time, range_, velocity_command

def model(k, X):
    return k[0]*X + k[1]

def linear_least_squares(velocity_command, D_step, T_step):
    N = len(T_step)
    Z = []
    k = zeros(2)
    for i in range(N):
        z = velocity_command[i] * T_step[i]
        Z.append(z)
    
    Z_vect = np.transpose(Z)
    X = np.transpose(np.array([ D_step, np.ones(len(D_step), dtype=int)]))

    delt, res, rank, s = np.linalg.lstsq(X, Z_vect)
    k += delt
    return k, Z

def mean(Z_meas, h_x):
    mean_V = sum(np.array(Z_meas) - np.array(h_x)) / len(h_x) 
    return mean_V

def variance(mean_V, V_noise):
    var_IR3_array = []

    for val in V_noise:
        var_IR3_array.append((val - mean_V) ** 2)


    var_V = sum(var_IR3_array) / len(var_IR3_array)
    return var_V

def PDF(var_V, mean_V):
    """ Work out the PDF's (mean and variance). """
    # Determine likelihoods.
    print("{}:{}\n".format(mean_V,var_V))

    f_v_IR3 = []
    x_array = linspace(-5,5,400)
    for val in x_array:
        f_v_IR3.append((1 / (2 * np.pi * var_V)) * np.exp((-1/2) * ((val - mean_V) ** 2) / (var_V)))

    return f_v_IR3

def main():
    #create time step arrray
    #solving ut = m(actual distance step) + c
    #aka     Z  = k[0](D_step) + k[1]
    #this goes into motion model 
    #Xn = Xn-1 + ut + Wn
    #need to find Wn

    time, range_, velocity_command = get_data()
    N = len(time)
    T_step = []
    R_step = []
    t_prev = 0
    r_prev = 0
    for i in range(N) :
        t_step = time[i] - t_prev
        T_step.append(t_step)
        t_prev = time[i]
        r_step = range_[i] - r_prev
        R_step.append(r_step)
        r_prev = range_[i]

    R_step = np.array(R_step)
    T_step = np.array(T_step)

    k, Z = linear_least_squares(velocity_command, R_step, T_step)
    h_x = model(k, R_step)
    
    V_noise = np.array(Z) - np.transpose(np.array(h_x))  

    mean_V = mean(Z, h_x)
    var_V = variance(mean_V, V_noise)
    f_v_MM = np.transpose(PDF(var_V, mean_V))


    plt.figure()
    x_array = linspace(-5,5,400)
    plt.plot(x_array, f_v_MM)

    plt.figure()
    plt.plot(time, R_step)
    plt.plot(time,h_x)


    plt.show()

        


main()