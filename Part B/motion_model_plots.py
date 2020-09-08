import numpy as np
import matplotlib.pyplot as plt

def plot_errors(error_x_array, error_y_array, error_theta_array):
    """ Plot the errors to determine the offset for the readings. """
    plt.subplot(131)
    plt.plot(error_x_array)
    plt.subplot(132)
    plt.plot(error_y_array)
    plt.subplot(133)
    plt.plot(error_theta_array)
    plt.show()