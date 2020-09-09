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

    #if odom_pose[0] != odom_pose_prev[0]:
    trajectory = arctan2((odom_pose[1] - odom_pose_prev[1]), (odom_pose[0] - odom_pose_prev[0]))
    #else:
        #trajectory = np.pi / 2

    d = sqrt(((odom_pose[1] - odom_pose_prev[1]) ** 2) + ((odom_pose[0] - odom_pose_prev[0]) ** 2)) 
    phi_1_local = angle_difference(odom_pose_prev[2], trajectory)
    phi_2_local = angle_difference(odom_pose[2], trajectory)
    
    difference_x = d * np.cos(odom_pose[2] + phi_1_local)#odom_pose[0] -odom_pose_prev[0]# + d * cos(odom_pose_prev[2] + phi_1_local)) # First column is x.
    difference_y = d * np.sin(odom_pose[2] + phi_1_local)#odom_pose[1] - odom_pose_prev[1]# + d * sin(odom_pose_prev[2] + phi_1_local)) # Second column is y.   
    difference_theta = (((odom_pose[2]  + phi_1_local + phi_2_local)) + np.pi) - np.pi#odom_pose[2] - odom_pose_prev[2]# + phi_1_local + phi_2_local) # Third colum is theta.
    #print(difference_x, difference_y, difference_theta)
    #difference_theta = min(2*np.pi - (odom_pose[2] - (odom_pose_prev[2] + phi_1_local + phi_2_local)),(odom_pose[2] - (odom_pose_prev[2] + phi_1_local + phi_2_local))) # Third colum is theta.  

    mu_x = 0 
    sigma_x = 0.03
    mu_y = 0
    sigma_y = 0.015
    mu_theta = 0
    sigma_theta = 0.03

    for m in range(M):
        # Currently is outputting odometry. Need to convert to global?? Add in the noise?
        particle_poses[m, 0] += difference_x + mu_x + randn() * sigma_x # First column is x.
        particle_poses[m, 1] += difference_y + mu_y + randn() * sigma_y# Second column is y.   
        particle_poses[m, 2] = (particle_poses[m, 2] + difference_theta) + mu_theta + randn() * sigma_theta # Third colum is theta.


    return particle_poses


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """ Apply sensor model and return particle weights. 
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

    r = np.sqrt((beacon_pose[0]) ** 2 + (beacon_pose[1]) ** 2)
    phi = arctan2(beacon_pose[1], beacon_pose[0])

    for m in range(M):
        r_m = np.sqrt((beacon_loc[0] - particle_poses[m][0]) ** 2 + (beacon_loc[1] - particle_poses[m][1]) ** 2) 
        phi_m = angle_difference(arctan2(beacon_loc[0] - particle_poses[m][0], beacon_loc[1] - particle_poses[m][1]), particle_poses[m][2])

        r_val = r - r_m
        phi_val = angle_difference(phi, phi_m)

        particle_weights[m] = gauss(r_val, sigma=0.1) * gauss(phi_val, sigma=0.2)


    return particle_weights
