###################################################################################################

""" Code setup. """

# Imports
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt, ones, zeros, linspace

import IR3_PSM
import sonar_PSM

###################################################################################################
def adjust_variance(zIR, varV_IR, zS1, varV_S1, zS2, varV_S2):
    #Piece wise to remove poor sensor readings readings
    if ( zIR < 0.5 or zIR > 3.0):
        varV_IR = 1000
    if (zS1 > 2.3):
        varV_S1 = 1000
    if (zS2 < 0.5):
        varV_S2 = 1000   
    return varV_IR, varV_S1, varV_S2 


def calibrate_Sensors():

    varV_IR3, kIR = IR3_PSM.calibration()
    varV_S1, varV_S2, kS1, kS2 = sonar_PSM.calibration()

    return varV_IR3, kIR, varV_S1, varV_S2, kS1, kS2 


def fuseMLE_sensors(zIR, zS1, zS2, x0, kIR, kS1, kS2, varV_IR3, varV_S1, varV_S2):
    xhat_S1, xhat_S2, varX_S1, varX_S2 = sonar_PSM.MLE_sonar(zS1, zS2, kS1, kS2, varV_S1, varV_S2)
    xhat_IR, varX_IR3 = IR3_PSM.linear_ML_IR(kIR, zIR, x0, varV_IR3)


    xhat_fusion, P = BLUE(varX_S1, varX_S2, varX_IR3, xhat_S1, xhat_S2, xhat_IR)
    return xhat_fusion, P

def BLUE(varS1, varS2, varIR3, S1_x, S2_x, IR3_x):
 
    #BLUE_Var = (1/(1/varS1 + 1/varS2 + 1/varIR3))
    P = (1/(1/varS1 + 1/varS2))
    #P = 1/(1/varS2)
    xhat_fusion =  ((1/varS1) * P *S1_x) + ((1/varS2) * P * S2_x)# +  (1/varIR3 * BLUE_Var * IR3_x)
    #xhat_fusion =  (1/varS2) * P * S2_x 
    return xhat_fusion, P

def test_Sensor_Fusion():
    data = loadtxt('Part A/calibration.csv', delimiter=',', skiprows=1)
    index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
    
    varV_IR3, kIR, varV_S1, varV_S2, kS1, kS2 = calibrate_Sensors()
    x0 = 5.0
    N = len(index)
    X_hat_fusion = []

    for n in range(N):
        zIR = raw_ir3[n]
        zS1 = sonar1[n]
        zS2 = sonar2[n]
        xhat_fusion, P= fuseMLE_sensors(zIR, zS1, zS2, x0, kIR, kS1, kS2, varV_IR3, varV_S1, varV_S2)
        X_hat_fusion.append(xhat_fusion)
        x0 = float(xhat_fusion)

    plt.figure()
    plt.plot(time, X_hat_fusion, 'ro')
    plt.plot(time, range_)

    plt.show()

test_Sensor_Fusion()