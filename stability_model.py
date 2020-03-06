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


def set_activation(th):
    '''
    :param th: ankle angle
    :return: [a_ta, a_s], activation constants for TA and soleus, respectively
    '''
    if th > np.pi/2:
        a_ta = ((1/0.06)*th - 26.1799)
        a_s = 0.01
    elif th < np.pi/2:
        a_s = (-5 * th + 5*np.pi/2)
        a_ta = 0.01
    else:
        a_ta = 0.01
        a_s = 0.01

    return [a_ta, a_s]

def dynamics(x, soleus, tibialis, control):
    """
    :param x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
    :param soleus: soleus muscle (HillTypeModel)
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :param control: True if balance should be controlled
    :return: derivative of state vector
    """

    # Muscle moment arms
    d_s = 0.05
    d_ta = 0.03

    # Muscle torques
    tau_s = d_s * soleus.get_force(soleus_length(x[0]), x[2])
    tau_ta = d_ta * tibialis.get_force(tibialis_length(x[0]), x[3])
    i_ankle = 90  # Inertia about the ankle

    # Activation constant
    if not control:
        a_s = 0.05
        a_ta = 0.4
    else:
        # Question 5: The control law can set the activation of each muscle
        # as a function of the system’s state variables
        a_ta = set_activation(x[0])[0]
        a_s = set_activation(x[0])[1]

    xdot = []
    xdot_1 = x[1]
    xdot_2 = ((tau_s - tau_ta + gravity_moment(x[0]))/i_ankle)
    xdot_3 = get_velocity(a_s, x[2], soleus.norm_tendon_length(soleus_length(x[0]),x[2]))
    xdot_4 = get_velocity(a_ta, x[3], tibialis.norm_tendon_length(tibialis_length(x[0]),x[3]))

    xdot.append(xdot_1)
    xdot.append(xdot_2)
    xdot.append(xdot_3)
    xdot.append(xdot_4)

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
    time = sol.t
    theta = sol.y[0,:]
    soleus_norm_length_muscle = sol.y[2,:]
    tibialis_norm_length_muscle = sol.y[3,:]

    soleus_moment_arm = .05
    tibialis_moment_arm = .03
    soleus_moment = []
    tibialis_moment = []
    for th, ls, lt in zip(theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
        soleus_moment.append(soleus_moment_arm * soleus.get_force(soleus_length(th), ls))
        tibialis_moment.append(-tibialis_moment_arm * tibialis.get_force(tibialis_length(th), lt))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, sol.y[0,:])
    plt.xlabel('Time (s)')
    plt.ylabel('Body angle (rad)')
    plt.title('Postural stablity body angle simulation over 10 seconds, controlled')
    plt.subplot(2,1,2)
    plt.title('Simulation of muscle torque produced by the soleus, TA, and gravity over 10 seconds, controlled')
    plt.plot(time, soleus_moment, 'r')
    plt.plot(time, tibialis_moment, 'g')
    plt.plot(time, gravity_moment(sol.y[0,:]), 'k')
    plt.legend(('soleus', 'tibialis', 'gravity'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torques (Nm)')
    plt.tight_layout()
    plt.show()


########################################################################################################################
if __name__ == "__main__":

    # Question 4
    simulate(False, 5)

    # Question 5
    simulate(True, 10)
