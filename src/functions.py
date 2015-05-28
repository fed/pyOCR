import math
import scipy as sp


def linear(x):
    return x


def d_linear(x):
    return 1


def tanh(x):
    return math.tanh(x)


def d_tanh(x):
    y = x
    return (1 - y) * (1 + y)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoid(x):
    y = x
    return y * (1 - y)


def error_mean_cuadratic(t, y):
    try:
        x = 0.5 * (t - y) ** 2
    except:
        print t, y
    return x

v_error_mean_cuadratic = sp.vectorize(error_mean_cuadratic)


def error_pattern(network, x, t):
    return sp.mean(v_error_mean_cuadratic(t, network.activate(x)))


def error_avg(network, training_set):
    return sp.mean(sp.array([error_pattern(network, x, t)
                             for x, t in training_set]))


functions = {'linear': (linear, d_linear),
             'sigmoid': (sigmoid, d_sigmoid),
             'tanh': (tanh, d_tanh)
             }
