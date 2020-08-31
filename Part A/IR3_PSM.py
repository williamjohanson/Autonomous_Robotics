""" Infrared 3 Probability Sensor Model (PSM) Code for ENMT482 assignment - Part A. """

###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, zeros, linspace
from numpy.linalg import lstsq
#from matplotlib.pyplot import figure, show, savefig
###################################################################################################

def model(x, k):
    return k[0] + k[1] / (x + k[2])

def model_nonlinear_least_squares_fit(r, x, iterations=5):

    N = len(r)
    A = ones((N, 3))
    k = zeros(3)

    for i in range(iterations):
    # Calculate Jacobians for current estimate of parameters.
        for n in range(N):
            A[n, 1] = 1 / (x[n] + k[2])
            A[n, 2] = -k[1] / (x[n] + k[2])**2

        # Use least squares to estimate the parameters.
        deltak, res, rank, s = lstsq(A, r - model(x, k))
        k += deltak
        print(k)
    return k



###################################################################################################

def IR_filter(self, parameter_list):
    pass

def PDF(h_x, Z_vector, V_vector, x_raw):
    """ Work out the PDF's (mean and variance). """
    # Determine likelihoods.
    mean_V = sum(np.array(Z_vector) - np.array(h_x)) / len(h_x)
    var_IR3_array = []

    for val in V_vector:
        var_IR3_array.append((val - mean_V) ** 2)


    var_V = sum(var_IR3_array) / len(var_IR3_array)

    print("{}:{}\n".format(mean_V,var_V))

    f_v_IR3 = []
    x_array = linspace(-5,5,400)
    for val in x_array:
        f_v_IR3.append((1 / (2 * np.pi * var_V)) * np.exp((-1/2) * ((val - mean_V) ** 2) / (var_V)))

    return f_v_IR3, 
    

def filter_outliers(V_vector):
    #using the Iglewics and Hoaglin's modified Z-score. This method requires a model for it to be performed.
    #The function requires the error vector and the median of the error.
    V_vector.sort()
    median = V_vector[(len(V_vector))/2]
    
    pass


def main():
    data = loadtxt('Part A/calibration.csv', delimiter=',', skiprows=1)
    index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    
    x_raw = raw_ir3[1:1500]
    Z_vector = range_[1:1500]

    k = model_nonlinear_least_squares_fit(x_raw, Z_vector)

    h_x = model(Z_vector, k)
    print(h_x)
    V_vector = np.array(Z_vector) - np.transpose(np.array(h_x))  
 
    f_v_IR3 = np.transpose(PDF(h_x, Z_vector, V_vector, x_raw))
   
    plt.figure()
    x_array = linspace(-5,5,400)

    plt.plot(x_array, f_v_IR3)
    plt.xlabel('IR3 voltage')
    plt.ylabel('PDF')
    plt.title('IR3 PDF') 

    plt.figure()
    plt.plot(Z_vector, x_raw, 'bo')
    plt.plot(h_x, x_raw,'k', linewidth=2)
    plt.xlabel('Distance (m)')
    plt.ylabel('Voltage (V)')
    plt.title('$k_1$ = %.3f, $k_2$ = %.3f, $k_3$ = %.3f' % (k[0], k[1], k[2]))
    plt.grid(True)

    plt.show()
    #savefig(__file__.replace('.py', '.pgf'), bbox_inches='tight')



main()