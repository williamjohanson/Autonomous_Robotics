""" Infrared 3 Probability Sensor Model (PSM) Code for ENMT482 assignment - Part A. """

###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, zeros, linspace
from numpy.linalg import lstsq

###################################################################################################

def model_h_x(x, k):
    return k[0] + k[1]/ (x + k[2]) 

def model_dh_x(x, k):
    return - k[1]/(x + k[2])**2

def model_nonlinear_least_squares_fit(x, z, iterations=5):
    
    N = len(z)
    A = ones((N, 3))
    k = zeros(3)
    
    for i in range(iterations):
    # Calculate Jacobians for current estimate of parameters.
        for n in range(N):
            A[n, 1] = 1 / (x[n] + k[2])
            A[n, 2] = -k[1] / (x[n] + k[2])**2
    
        # Use least squares to estimate the parameters.
        deltak, res, rank, s = lstsq(A, z - model_h_x(x, k))
        k += deltak
        print(k)
    return k


###################################################################################################

def linear_ML_IR(k, z, x0, var_V):

    h_x = model_h_x(x0, k)
    dh_x = model_dh_x(x0, k) 
    x_hat = (z - h_x) / dh_x + x0
    var_x_hat = var_V/(dh_x)**2
    return x_hat, var_x_hat

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
    

def filter_outliers(V_vector):
    #using the Iglewics and Hoaglin's modified Z-score. This method requires a model for it to be performed.
    #The function requires the error vector and the median of the error.
    V_vector.sort()
    median = V_vector[(len(V_vector))/2]
    
    pass

def BLUE(varS1, varS2, varIR3, S1_x, S2_x, IR3_x):
    x_hat_fusion =  ((1/varS2)*S1_x + (1/varS2)*S2 + (1/varIR3)*IR3_x) / (1/varS1 + 1/varS2 + 1/varIR3)
    return x_hat_fusion

def main():
    data = loadtxt('Part A/calibration.csv', delimiter=',', skiprows=1)
    index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    
    Z_meas = raw_ir3[1:758]
    X_state = range_[1:758]

    k = model_nonlinear_least_squares_fit(X_state, Z_meas)
    X_array = linspace(0.1,0.8,201)
    h_x = model_h_x(X_state, k)
    h_x_plot = model_h_x(X_array, k)

    V_noise = np.array(Z_meas) - np.transpose(np.array(h_x))  
 
    mean_V = mean(Z_meas, h_x)
    var_V = variance(mean_V, V_noise)
    f_v_IR3 = np.transpose(PDF(var_V, mean_V))
    
    #for a given measurement z this function will determine where the ML estimate of the next point is, but
    #where do i get my initial guess from? is it form my other sensors?
    x0 = X_state[1]
    N = len(Z_meas)
    X_hat_array = []
    for n in range(N):
        z = Z_meas[n]
        var_z = V_noise[n]
        x_hat, var_x_hat = linear_ML_IR(k, z, x0, var_V)
        X_hat_array.append(x_hat)
        x0 = x_hat


    plt.figure()
    x_array = linspace(-5,5,400)

    plt.plot(x_array, f_v_IR3)
    plt.xlabel('IR3 voltage')
    plt.ylabel('PDF')
    plt.title('IR3 PDF') 

    plt.figure()
    plt.plot(X_state, Z_meas, 'bo')
    plt.plot(X_array, h_x_plot,'ko')
    plt.plot(X_hat_array, Z_meas, 'ro')
    plt.ylabel('Distance (m)')
    plt.xlabel('Voltage (V)')
    plt.title('$k_1$ = %.3f, $k_2$ = %.3f, $k_3$ = %.3f' % (k[0], k[1], k[2]))
    plt.grid(True)

    plt.show()
    #savefig(__file__.replace('.py', '.pgf'), bbox_inches='tight')



main()