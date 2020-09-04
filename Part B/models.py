"""Particle filter sensor and motion model implementations.

M.P. Hayes and M.J. Edwards,
Department of Electrical and Computer Engineering
University of Canterbury
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference


def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """

    M = particle_poses.shape[0]
    trajectory = np.arctan((odom_pose[1] - odom_pose_prev[1]) / (odom_pose[0] - odom_pose_prev[0]))
    d = np.sqrt((odom_pose[1] - odom_pose_prev[1]) ** 2 + (odom_pose[0] - odom_pose_prev[0]) ** 2)
    phi_1_local = odom_pose_prev[2] - trajectory
    phi_2_local = odom_pose[2] - trajectory

    #delta_theta = odom_pose
    
    #x_new = current_particle_poses[0] - current_particle_poses[1] * sin()
    
    # TODO.  For each particle calculate its predicted pose plus some
    # additive error to represent the process noise.  With this demo
    # code, the particles move in the -y direction with some Gaussian
    # additive noise in the x direction.  Hint, to start with do not
    # add much noise.
    '''
    for m in range(M):
        particle_poses[m, 0] = odom_pose_prev[0] -(speed_command[0] / speed_command[1]) * sin(odom_pose_prev[2]) + (speed_command[0] / speed_command[1]) * sin(odom_pose_prev[2] + speed_command[1] * dt)
        particle_poses[m, 1] = odom_pose_prev[1] + (speed_command[0] / speed_command[1]) * cos(odom_pose_prev[2]) - (speed_command[0] / speed_command[1]) * cos(odom_pose_prev[2] + speed_command[1] * dt)
        particle_poses[m, 2] = odom_pose_prev[2] + speed_command[1] * dt
    
    return particle_poses
    '''    
    
    for m in range(M):
        # Currently is outputting odometry. Need to convert to global?? Add in the noise?
        particle_poses[m, 0] += d * cos(odom_pose_prev[2] + phi_1_local)# First column is x.
        particle_poses[m, 1] += d * sin(odom_pose_prev[2] + phi_1_local) # Second column is y.   
        particle_poses[m, 2] += particle_poses[m, 2] + phi_1_local + phi_2_local# - odom_pose_prev[2] # Third colum is theta.
    
    return particle_poses
    
'''

    for m in range(M):
        particle_poses[m, 0] += randn(1) * 0.5
        particle_poses[m, 1] -= 0.1
    
    return particle_poses'''


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)

    r_meas = []
    phi_meas = []
    r_true = []
    phi_true = []

    # Bug in the error being so damn small. Something to do with my particle poses??
    for i in range(len(particle_poses)):
        r_meas.append(np.sqrt((beacon_pose[0] - particle_poses[i][0]) ** 2 + (beacon_pose[1] - particle_poses[i][1]) ** 2))
        theta_1 = np.arctan2(beacon_pose[1] - particle_poses[i][1], beacon_pose[0] - particle_poses[i][0])
        theta_2 = particle_poses[i][2]
        phi_meas.append(min(2*np.pi - (theta_1 - theta_2), theta_1 - theta_2))

        r_true.append(np.sqrt((beacon_loc[0] - particle_poses[i][0]) ** 2 + (beacon_loc[1] - particle_poses[i][1]) ** 2))
        theta_1_true = np.arctan2(beacon_loc[1] - particle_poses[i][1], beacon_loc[0] - particle_poses[i][0])
        theta_2_true = particle_poses[i][2]
        phi_true.append(min(2*np.pi - (theta_1_true - theta_2_true), theta_1_true - theta_2_true))

    error_r_array = []
    error_phi_array = []

    for i in range(len(r_meas)):    
        error_r_array.append(r_true[i] - r_meas[i])
        error_phi_array.append(min(2*np.pi - (phi_true[i] - phi_meas[i]), phi_true[i] - phi_meas[i]))

    mean_r = sum(error_r_array) / len(error_r_array)
    mean_phi = sum(error_phi_array) / len(error_phi_array)

    var_r_array = []
    var_phi_array = []

    

    for error_r in error_r_array:
        var_r_array.append((error_r - mean_r) ** 2)

    for error_phi in error_phi_array:
        var_phi_array.append(min(2*np.pi - (error_phi - mean_phi), error_phi - mean_phi) ** 2)
    
    var_r = sum(var_r_array) / len(var_r_array)
    var_phi = sum(var_phi_array) / len(var_phi_array)
    
    print(var_r, var_phi)
    
    for m in range(M):

        particle_weights[m] = (1 / np.sqrt(2 * np.pi * var_r)) * np.exp((-1/2) * ((error_r_array[m] - mean_r) ** 2) / (var_r)) * (1 / np.sqrt(2 * np.pi * var_phi)) * np.exp((-1/2) * ((error_phi_array[m] - mean_phi) ** 2) / (var_phi))#np.random.normal(mean_r, np.sqrt(var_r), 1) * np.random.normal(mean_phi, np.sqrt(var_phi), 1)


    return particle_weights
