import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp

class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments.
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to
            normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to
            normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """

        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))


def damped_equilibrium(vm, *musc_arg):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    alpha = 0
    a, lm, lt, beta = musc_arg
    f_lm = force_length_muscle(lm)
    f_vm = force_velocity_muscle(vm)
    f_lp = force_length_parallel(lm)
    f_lt = force_length_tendon(lt)
    return (((a * f_lm * f_vm) + f_lp + (beta*vm)) * np.cos(alpha)) - f_lt


def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """
    beta = 0.1  # damping coefficient (see damped model in Millard et al.)

    # WRITE CODE HERE TO CALCULATE VELOCITY`
    vnorm0 = 0.0
    vnorm = fsolve(damped_equilibrium, np.array([vnorm0]), args=(a, lm, lt, beta))
    return vnorm


def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return ft: normalized tension produced by tendon
    """

    lts = 1.0  # slack length of tendon (SE)
    t_norm = []

    if type(lt) is int or type(lt) is float or type(lt) is np.float64:
        lt = np.array([lt])

    for i in range(len(lt)):
        if lt[i] < lts:
            t_norm.append(0)
        else:
            t_norm.append((10.0 * (lt[i] - lts)) + (240.0 * (lt[i] - lts)**2))

    return np.array(t_norm)


def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """

    lpes = 1.0  # slack length of PE
    f_norm = []

    if type(lm) is int or type(lm) is float or type(lm) is np.float64:
        lm = np.array([lm])

    for i in range(len(lm)):
        if lm[i] < lpes:
            f_norm.append(0)
        else:
            f_norm.append((3.0 * (lm[i] - lpes)**2) / (.6 + lm[i] - lpes))

    return np.array((f_norm))


def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.subplot(2,1,1)
    plt.plot(lm, force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized muscle velocity')
    plt.ylabel('Force scale factor')
    plt.tight_layout()
    plt.show()


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)


class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.abc.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The sampples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.

    WRITE CODE HERE 1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions. 
    """
    data = np.array([
        [39.44272446,    3.529411765],
        [41.79566563,    1.882352941],
        [37.3993808 ,    9.882352941],
        [38.39009288,    14.58823529],
        [41.42414861,    14.58823529],
        [41.42414861,    15.76470588],
        [40.43343653,    17.64705882],
        [40.37151703,    21.17647059],
        [39.25696594,    24.23529412],
        [41.42414861,    26.58823529],
        [41.3622291 ,    31.76470588],
        [42.04334365,    32],
        [40.43343653,    36.70588235],
        [43.40557276,    34.82352941],
        [42.91021672,    23.52941176],
        [43.40557276,    23.52941176],
        [43.83900929,    22.11764706],
        [45.57275542,    43.52941176],
        [45.57275542,    46.35294118],
        [46.6873065 ,    44.94117647],
        [44.39628483,    45.41176471],
        [43.40557276,    44.70588235],
        [42.53869969,    41.88235294],
        [42.84829721,    46.35294118],
        [42.84829721,    48],
        [43.15789474,    50.11764706],
        [43.46749226,    53.88235294],
        [43.71517028,    56.94117647],
        [44.39628483,    60.47058824],
        [46.43962848,    62.35294118],
        [47.43034056,    62.35294118],
        [47.73993808,    66.58823529],
        [48.97832817,    62.82352941],
        [45.69659443,    67.52941176],
        [45.94427245,    70.58823529],
        [46.43962848,    71.52941176],
        [46.43962848,    73.64705882],
        [46.6873065 ,    75.29411765],
        [47.43034056,    71.52941176],
        [47.12074303,    80.23529412],
        [47.55417957,    81.41176471],
        [47.43034056,    81.17647059],
        [48.17337461,    83.05882353],
        [48.91640867,    84.94117647],
        [48.97832817,    81.64705882],
        [49.78328173,    81.41176471],
        [49.78328173,    84.70588235],
        [49.65944272,    86.35294118],
        [50.27863777,    87.29411765],
        [50.6501548,    86.82352941],
        [50.6501548,    84.70588235],
        [50.95975232,    79.76470588],
        [50.77399381,    77.64705882],
        [51.20743034,    78.11764706],
        [50.58823529,    74.11764706],
        [49.41176471,    75.76470588],
        [53.56037152,    78.58823529],
        [53.56037152,    83.29411765],
        [53.25077399,    88.47058824],
        [52.63157895,    88.94117647],
        [51.51702786,    89.41176471],
        [51.70278638,    90.58823529],
        [50.6501548 ,    90.35294118],
        [45.3869969 ,    53.64705882],
        [45.26315789,    54.11764706],
        [53.62229102,    92.23529412],
        [53.93188854,    91.76470588],
        [54.30340557,    94.11764706],
        [54.11764706,    93.64705882],
        [53.49845201,    94.35294118],
        [53.56037152,    96.23529412],
        [53.86996904,    96.47058824],
        [54.6749226 ,    99.29411765],
        [55.72755418,    95.76470588],
        [56.16099071,    96],
        [56.47058824,    99.52941176],
        [56.84210526,    99.29411765],
        [57.2755418 ,    97.64705882],
        [57.08978328,    99.52941176],
        [57.52321981,    99.29411765],
        [57.83281734,    99.29411765],
        [58.57585139,    96],
        [57.83281734,    91.05882353],
        [58.45201238,    90.58823529],
        [59.38080495,    90.82352941],
        [59.69040248,    95.52941176],
        [59.31888545,    97.64705882],
        [58.88544892,    98.82352941],
        [59.81424149,    96.70588235],
        [60,             95.52941176],
        [60.55727554,    99.29411765],
        [60.61919505,    93.17647059],
        [61.3622291 ,    94.82352941],
        [62.35294118,    96.23529412],
        [61.3622291 ,    91.52941176],
        [61.42414861,    87.29411765],
        [62.22910217,    88.94117647],
        [61.3003096 ,    84.47058824],
        [61.42414861,    79.52941176],
        [61.42414861,    76.70588235],
        [62.6006192 ,    79.52941176],
        [63.46749226,    79.52941176],
        [63.83900929,    80],
        [63.40557276,    80.70588235],
        [64.52012384,    81.64705882],
        [63.65325077,    85.64705882],
        [63.15789474,    85.88235294],
        [63.77708978,    86.58823529],
        [63.71517028,    89.41176471],
        [64.52012384,    86.58823529],
        [63.9628483 ,    76],
        [65.13931889,    76],
        [66.43962848,    75.29411765],
        [65.63467492,    72.47058824],
        [66.13003096,    72],
        [65.75851393,    68],
        [65.75851393,    66.35294118],
        [65.3869969 ,    64],
        [66.87306502,    66.11764706],
        [67.24458204,    63.05882353],
        [67.73993808,    62.35294118],
        [68.42105263,    59.52941176],
        [67.55417957,    51.76470588],
        [63.40557276,    59.29411765],
        [63.40557276,    52.70588235],
        [64.76780186,    53.64705882],
        [64.76780186,    52],
        [65.75851393,    48],
        [67.05882353,    42.58823529],
        [67.43034056,    35.76470588],
        [69.41176471,    41.64705882],
        [70.58823529,    48.47058824],
        [71.45510836,    34.58823529],
        [73.43653251,    34.58823529],
        [68.54489164,    27.52941176],
        [70.21671827,    29.64705882],
        [70.46439628,    29.64705882],
        [72.44582043,    24.70588235],
        [73.00309598,    25.41176471],
        [73.43653251,    18.58823529],
        [73.43653251,    17.64705882],
        [73.43653251,    12.70588235],
        [74.42724458,    12.94117647],
        [75.23219814,    17.88235294],
        [75.41795666,    12.70588235],
        [76.40866873,    8.705882353],
        [70.40247678,    49.64705882]
    ])

    length = data[:,0]
    tension = data[:,1]
    max_tension = max(tension)
    length_idx = np.argmax(tension)
    length_max_tension = length[length_idx]
    norm_length = length/length_max_tension
    norm_tension = tension/max_tension

    centres = np.arange(min(norm_length)+0.1, max(norm_length), .2)
    width = .15
    result = Regression(norm_length, norm_tension, centres, width, .1, sigmoids=False)

    return result

force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()


def force_length_muscle(lm):
    """
    :param lm: muscle (contractile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))


########################################################################################################################
if __name__ == "__main__":
    # Question 1
    plot_curves()

    # Question 2
    print(get_velocity(1,1,1.01))

    # Question 3
    f0M = 100.0
    rml = 0.3
    rtl = 0.1
    total_length = rml + rtl
    muscle = HillTypeMuscle(f0M, rml, rtl)


    def f(t,x):
        if t < 0.5:
            a = 0
        else:
            a = 1

        # the derivative of normalized muscle length is normalized muscle velocity
        vnorm = get_velocity(a, x, muscle.norm_tendon_length(total_length,x))
        return vnorm


    sol = solve_ivp(f, [0, 2], np.array([1.0]), max_step=0.01, rtol=1e-3, atol=1e-6)
    ce_force = muscle.get_force(total_length, sol.y.T)

    plt.subplot(2,1,1)
    plt.plot(sol.t, sol.y.T)
    plt.xlabel('Time (s)')
    plt.ylabel('CE Length (m)')
    plt.subplot(2,1,2)
    plt.plot(sol.t, ce_force, 'g')
    plt.xlabel('Time (s)')
    plt.ylabel('CE Force (N)')
    plt.tight_layout()
    plt.show()





