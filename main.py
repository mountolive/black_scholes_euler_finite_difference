# Entrancy point
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from black_scholes_finite_difference.payoff import PayoffCall
from black_scholes_finite_difference.option import VanillaOption
from black_scholes_finite_difference.pde import BlackScholesPDE
from black_scholes_finite_difference.finite_diff_method import FDMEuler
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

if __name__ == '__main__':
    k = 0.5
    r = 0.05
    # Volatility of the underlying asset
    v = 0.2
    T = 1.0
    x_dom = 1.0
    j = 20
    t_dom = T
    n = 20
    payoff_call = PayoffCall(k)
    call_option = VanillaOption(payoff_call, k, r, T, v)
    bs_pde = BlackScholesPDE(call_option)
    fdm_euler = FDMEuler(x_dom, t_dom, j, n, bs_pde)
    # Running

    fdm_euler.step_march()

    # Graphication

    x, y, z = np.loadtxt('fdm_result.csv', unpack=True)

    X = np.reshape(x, (20, 20))
    Y = np.reshape(y, (20, 20))
    Z = np.reshape(z, (20, 20))
    step = 0.04
    maxval = 1.0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.YlGnBu_r)
    ax.set_zlim3d(0, 1.0)
    ax.set_xlabel(r'$S$')
    ax.set_ylabel(r'$T-t$')
    ax.set_zlabel(r'$C(S,t)$')
    plt.show()


