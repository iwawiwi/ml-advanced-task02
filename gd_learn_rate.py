__author__ = 'iwawiwi'

import numpy as np
from time import time
import grad_descent as reg
import matplotlib.pyplot as plt

n_sample_list = [10000, 5000, 1000, 500, 100, 50, 10]
n_gd_iter_list = [15000, 7500, 1500, 750, 150, 75, 15]

# Orde
order = 3
# Number of Iteration
# n_gd_iter = 15000
# Learning rate
gd_alpha1 = 0.00010
# gd_alpha2 = 0.00015
# gd_alpha3 = 0.000171
gd_alpha2 = 0.00015
gd_alpha3 = 0.00021
# Stocasthic GD sample to take
sgd2_sample_size = 3


for n_sample in n_sample_list:
    for n_gd_iter in n_gd_iter_list:
        # print 'LEN X = ', len(x)
        # true signal
        x = np.linspace(0.0, 2 * np.pi, 100, True)
        y_true = np.sin(x)

        # noisy signal
        # n_sample = 10000
        x_sample = np.linspace(0.0, 2 * np.pi, n_sample, True)
        noise = np.random.normal(0, 0.15, len(x_sample))
        y_sample = np.sin(x_sample) + noise

        # Standard Regression
        t0 = time()
        w_stand, J_stand = reg.standReg(x_sample, y_sample, order)
        # Gradient Descent alpha 0.0001
        t1 = time()
        w_gd, J_gd_history = reg.gradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha1)
        print 'GD 1 finished'
        # Gradient Descent alpha 0.001
        t2 = time()
        w_gd2, J_gd_history2 = reg.gradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha2)
        print 'GD 2 finished'
        # Gradient Descent alpha 0.01
        t3 = time()
        w_gd3, J_gd_history3 = reg.gradDescent(x_sample, y_sample, order, n_gd_iter, gd_alpha3)
        print 'GD 3 Finished'
        t4 = time()

        # compute time
        ct_normal_eq = t1 - t0
        ct_gd_1 = t2 - t1
        ct_gd_2 = t3 - t2
        ct_gd_3 = t4 - t2

        print 'X before model created = ', x

        # create model for drawing
        y_model_stand = reg.createModel(x, w_stand)
        y_model_gd1 = reg.createModel(x, w_gd)
        y_model_gd2 = reg.createModel(x, w_gd2)
        y_model_gd3 = reg.createModel(x, w_gd3)

        # plot results
        plt_title = '[N' + str(n_sample) + '][M' + str(order) + '][iter' + \
                    str(n_gd_iter) + ']GRAPH'
        fig = plt.figure(plt_title)
        ax = fig.add_subplot(111)
        ax.plot(x, y_true, 'g-', linewidth=2, label='True')
        ax.scatter(x_sample, y_sample, s=50, facecolors='none', edgecolors='b', linewidths=0.5, label='Data')
        ax.plot(x, y_model_stand, 'r--', linewidth=2, label='Standard')  # plot LR Standard
        ax.plot(x, y_model_gd1, 'g--', linewidth=2, label='GD' + str(gd_alpha1))  # Plot GD1
        ax.plot(x, y_model_gd2, 'm--', linewidth=2, label='GD' + str(gd_alpha2))  # Plot GD2
        ax.plot(x, y_model_gd3, 'c--', linewidth=2, label='GD' + str(gd_alpha3))  # Plot GD3
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Regression with M = ' + str(order) + ', N = ' + str(len(x_sample)))
        plt.legend()
        plt.grid()

        # print 'X before plot = ', x

        # plot computation time comparison
        plt_title2 = '[N' + str(n_sample) + '][M' + str(order) + '][iter' + \
                     str(n_gd_iter) + ']COMPTIME'
        fig = plt.figure(plt_title2)
        ax = fig.add_subplot(111)
        cts = [ct_normal_eq, ct_gd_1, ct_gd_2, ct_gd_3]
        b = [0.15, 0.35, 0.55, 0.75]
        plt.xlim(0.0, 1.0)
        tick_offset = [0.05] * 4
        xticks = [x + y for x, y in zip(b, tick_offset)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(('NE', 'GD1', 'GD2', 'GD3'))
        ax.bar(b, cts, width=0.1, color='r')
        plt.xlabel('Methods')
        plt.ylabel('Time (s)')
        plt.title('Computation Time of with Iter = ' + str(n_gd_iter))
        plt.grid()

        # print 'X last = ', x

plt.show()