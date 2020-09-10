"""Particle filter sensor and motion model implementations.

            William Johanson
            University of Canterbury
"""
# Imports.
import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference

###################################################################################################

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

    # Robot Trajectory between poses.
    trajectory = arctan2((odom_pose[1] - odom_pose_prev[1]), (odom_pose[0] - odom_pose_prev[0]))

    # Pose variables.
    d = sqrt(((odom_pose[1] - odom_pose_prev[1]) ** 2) + ((odom_pose[0] - odom_pose_prev[0]) ** 2)) 
    phi_1_local = angle_difference(odom_pose_prev[2], trajectory)
    phi_2_local = angle_difference(odom_pose[2], trajectory)
    
    # Calculate difference between poses.
    difference_x = d * cos(odom_pose[2] + phi_1_local) # First column is x.
    difference_y = d * sin(odom_pose[2] + phi_1_local) # Second column is y.   
    difference_theta = wraptopi(phi_1_local + phi_2_local) # Third colum is theta.
    
    # Assign gaussian noise values.
    mu_x = 0
    sigma_x = 0.01
    mu_y = 0
    sigma_y = 0.008
    mu_theta = 0
    sigma_theta = 0.0007

    # Update particle poses.
    for m in range(M):
        particle_poses[m, 0] += difference_x + mu_x + randn() * sigma_x # First column is x.
        particle_poses[m, 1] += difference_y + mu_y + randn() * sigma_y # Second column is y.   
        particle_poses[m, 2] = wraptopi((particle_poses[m, 2] + difference_theta) + mu_theta + randn() * sigma_theta) # Third colum is theta.


    return particle_poses

###################################################################################################

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

    # Calculate beacon measured from camera frame.
    r = np.sqrt((beacon_pose[0]) ** 2 + (beacon_pose[1]) ** 2)
    phi = arctan2(beacon_pose[1], beacon_pose[0])

    for m in range(M):
        # Calculate the true location of beacon from particle pose m.
        r_m = np.sqrt((beacon_loc[0] - particle_poses[m][0]) ** 2 + (beacon_loc[1] - particle_poses[m][1]) ** 2) 
        phi_m = angle_difference(arctan2(beacon_loc[0] - particle_poses[m][0], beacon_loc[1] - particle_poses[m][1]), particle_poses[m][2])

        # Calculate error.
        r_val = r - r_m
        phi_val = angle_difference(phi, phi_m)

        # Update particle weights.
        particle_weights[m] = gauss(r_val, mu=0, sigma=(r_m ** 2)) * gauss(phi_val, mu=0, sigma=(r_m ** 2))


    return particle_weights
