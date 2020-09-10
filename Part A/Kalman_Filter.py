###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, zeros, linspace
from numpy.linalg import lstsq

import IR3_PSM
import sonar_PSM
import MotionModel
import Sensor_Fusion
###################################################################################################
def calculate_Prior(z, x0, k, var_V):
    #xp = x0 + b
    
    b, var_X = MotionModel.MLE_D_step(k, z, var_V)
    print(b)
    P1_prior = var_X
    x_prior = x0 + b
    return x_prior, P1_prior


def Kalman_Filter():
    #data = loadtxt('Part A/calibration.csv', delimiter=',', skiprows=1)
    #index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    data = loadtxt('Part A/test.csv', delimiter=',', skiprows=1)
    index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    #data = loadtxt('Part A/training1.csv', delimiter=',', skiprows=1)
    #index, range_, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    N = len(time)
    t_prev = 0
    x_prev = 0.0
    k, var_V = MotionModel.calibrate()
    X_prior_array = []
    X_posterior_array = []
    X_hat_fusion = []
    K_gain_array = []


    varV_IR3, kIR, varV_S1, varV_S2, kS1, kS2 = Sensor_Fusion.calibrate_Sensors()
    print(varV_IR3,varV_S1, varV_S2)
    varV_IR3_const = varV_IR3 
    varV_S1_const = varV_S1
    varV_S2_const = varV_S2
    x0 = 0.3
    N = len(index)
   

    for i in range(N):
        zMM = velocity_command[i] * (time[i] - t_prev)
        x_prior, P_prior = calculate_Prior(zMM, x_prev, k, var_V)
        X_prior_array.append(x_prior)

        #varV_IR3 = varV_IR3_const 
        #varV_S1 = varV_S1_const 
        #varV_S2 = varV_S2_const 
        zIR = raw_ir3[i]
        zS1 = sonar1[i]
        zS2 = sonar2[i]

        #varV_IR, varV_S1, varV_S2 = Sensor_Fusion.adjust_variance(zIR, varV_IR3, zS1, varV_S1, zS2, varV_S2)

        xhat_fusion, P = Sensor_Fusion.fuseMLE_sensors(zIR, zS1, zS2, x0, kIR, kS1, kS2, varV_IR3, varV_S1, varV_S2)
        X_hat_fusion.append(xhat_fusion)
        x0 = float(xhat_fusion)

        #print(P, P_prior, "wo")
        #K_gain = 1/P /  (1/P + 1/P_prior)
        #print(K_gain)
        #K_gain_array.append(K_gain)

        #x_posterior = K_gain*x_prior + (1 - K_gain)*xhat_fusion 
        x_posterior = 0.5*x_prior + (1 - 0.5)*xhat_fusion 
        #x_posterior = x_prior
        X_posterior_array.append(x_posterior)
        x_prev = x_posterior
        t_prev = time[i]

    plt.figure()
    plt.plot(time, X_posterior_array, 'ro', alpha=0.2)
    plt.plot(time, X_prior_array, 'ko', alpha=0.2)
    plt.plot(time, X_hat_fusion, 'bo', alpha=0.2)

    #plt.figure()
    #plt.plot(time, K_gain_array)

    plt.show()

Kalman_Filter()   