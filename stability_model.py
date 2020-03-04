"""
Simple model of standing postural stability, consisting of foot and body segments,
and two muscles that create moments about the ankles, tibialis anterior and soleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity


def soleus_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: soleus length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, .03])
    insertion = [-.05, -.02]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def tibialis_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def gravity_moment(theta):
    """
    :param theta: angle of body segment (up from prone)
    :return: moment about ankle due to force of gravity on body
    """
    mass = 75 # body mass (kg; excluding feet)
    centre_of_mass_distance = 1 # distance from ankle to body segment centre of mass (m)
    g = 9.81 # acceleration of gravity
    return mass * g * centre_of_mass_distance * np.sin(theta - np.pi / 2)


def dynamics(x, soleus, tibialis, control):
    """
    :param x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
    :param soleus: soleus muscle (HillTypeModel)
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :param control: True if balance should be controlled
    :return: derivative of state vector
    """

    # WRITE CODE HERE TO IMPLEMENT THE MODEL
    f_sM = soleus.f0M
    f_taM = tibialis.f0M
    d_s = 0.05
    d_ta = 0.03
    tau_s = f_sM * d_s * soleus.get_force(soleus.norm_tendon_length(soleus_length(x[0]), x[2]), x[2])[0]
    tau_ta = f_taM * d_ta * tibialis.get_force(tibialis.norm_tendon_length(tibialis_length(x[0]), x[3]), x[3])[0]
    a_s = 0.05
    a_ta = 0.4

    f_ext = 0
    d_ext = 0

    i_ankle = 90

    xdot = []

    xdot_1 = x[1]
    # f_ext*d_ext*np.cos(x[0] - np.pi/2)
    xdot_2 = ((tau_s - tau_ta + gravity_moment(x[0]))/i_ankle)
    xdot_3 = get_velocity(a_s, x[2], soleus.norm_tendon_length(soleus_length(x[0]),x[2]))
    # print(xdot_3)
    # xdot_4 = get_velocity(a_ta, x[3], tibialis.norm_tendon_length(tibialis_length(x[0]),x[3]))
    # print(xdot_4)

    xdot.append(xdot_1)
    # xdot.append(xdot_2)
    # xdot.append(30)

    # xdot.append(xdot_3)
    # xdot.append(xdot_4)
    xdot.append(30)
    xdot.append(30)
    xdot.append(30)
    return (xdot)


def simulate(control, T):
    """
    Runs a simulation of the model and plots results.
    :param control: True if balance should be controlled
    :param T: total time to simulate, in seconds
    """
    rest_length_soleus = soleus_length(np.pi/2)
    rest_length_tibialis = tibialis_length(np.pi/2)

    soleus = HillTypeMuscle(16000, .6*rest_length_soleus, .4*rest_length_soleus)
    tibialis = HillTypeMuscle(2000, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        return dynamics(x, soleus, tibialis, control)

    sol = solve_ivp(f, [0, T], [np.pi/2, 0, 1, 1], rtol=1e-5, atol=1e-8)
    print(sol)
    time = sol.t
    theta = sol.y[0,:]
    soleus_norm_length_muscle = sol.y[2,:]
    tibialis_norm_length_muscle = sol.y[3,:]

    soleus_moment_arm = .05
    tibialis_moment_arm = .03
    soleus_moment = []
    tibialis_moment = []
    for th, ls, lt in zip(theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
        soleus_moment.append(np.array([soleus_moment_arm]) * soleus.get_force(soleus_length(th), ls))
        tibialis_moment.append(np.array([-tibialis_moment_arm]) * tibialis.get_force(tibialis_length(th), lt))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, sol.y[0,:])
    plt.ylabel('Body angle (rad)')
    plt.subplot(2,1,2)
    plt.plot(time, soleus_moment, 'r')
    plt.plot(time, tibialis_moment, 'g')
    plt.plot(time, gravity_moment(sol.y[0,:]), 'k')
    plt.legend(('soleus', 'tibialis', 'gravity'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torques (Nm)')
    plt.tight_layout()
    plt.show()


simulate(False, 5)

