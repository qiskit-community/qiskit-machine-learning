# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
ad hoc dataset
"""

import numpy as np
import scipy
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def ad_hoc_data(training_size, test_size, n, gap, plot_data=False):
    """ returns ad hoc dataset """
    class_labels = [r'A', r'B']
    count = 0
    if n == 2:
        count = 100
    elif n == 3:
        count = 20   # coarseness of data separation

    label_train = np.zeros(2*(training_size+test_size))
    sample_train = []
    sample_a = [[0 for x in range(n)] for y in range(training_size+test_size)]
    sample_b = [[0 for x in range(n)] for y in range(training_size+test_size)]

    sample_total = [[[0 for x in range(count)] for y in range(count)] for z in range(count)]

    # interactions = np.transpose(np.array([[1, 0], [0, 1], [1, 1]]))

    steps = 2*np.pi/count

    # sx = np.array([[0, 1], [1, 0]])
    # X = np.asmatrix(sx)
    # sy = np.array([[0, -1j], [1j, 0]])
    # Y = np.asmatrix(sy)
    s_z = np.array([[1, 0], [0, -1]])
    z_m = np.asmatrix(s_z)
    j_m = np.array([[1, 0], [0, 1]])
    j_m = np.asmatrix(j_m)
    h_m = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    h_2 = np.kron(h_m, h_m)
    h_3 = np.kron(h_m, h_2)
    h_m = np.asmatrix(h_m)
    h_2 = np.asmatrix(h_2)
    h_3 = np.asmatrix(h_3)

    f_a = np.arange(2**n)

    my_array = [[0 for x in range(n)] for y in range(2**n)]

    for arindex, _ in enumerate(my_array):
        temp_f = bin(f_a[arindex])[2:].zfill(n)
        for findex in range(n):
            my_array[arindex][findex] = int(temp_f[findex])

    my_array = np.asarray(my_array)
    my_array = np.transpose(my_array)

    # Define decision functions
    maj = (-1)**(2*my_array.sum(axis=0) > n)
    parity = (-1)**(my_array.sum(axis=0))
    # dict1 = (-1)**(my_array[0])
    d_m = None
    if n == 2:
        d_m = np.diag(parity)
    elif n == 3:
        d_m = np.diag(maj)

    basis = np.random.random((2**n, 2**n)) + 1j*np.random.random((2**n, 2**n))
    basis = np.asmatrix(basis).getH()*np.asmatrix(basis)

    [s_a, u_a] = np.linalg.eig(basis)

    idx = s_a.argsort()[::-1]
    s_a = s_a[idx]
    u_a = u_a[:, idx]

    m_m = (np.asmatrix(u_a)).getH()*np.asmatrix(d_m)*np.asmatrix(u_a)

    psi_plus = np.transpose(np.ones(2))/np.sqrt(2)
    psi_0 = 1
    for k in range(n):
        psi_0 = np.kron(np.asmatrix(psi_0), np.asmatrix(psi_plus))

    sample_total_a = []
    sample_total_b = []
    sample_total_void = []
    if n == 2:
        for n_1 in range(count):
            for n_2 in range(count):
                x_1 = steps*n_1
                x_2 = steps*n_2
                phi = x_1*np.kron(z_m, j_m) + x_2*np.kron(j_m, z_m) + \
                    (np.pi-x_1)*(np.pi-x_2)*np.kron(z_m, z_m)
                u_u = scipy.linalg.expm(1j*phi)  # pylint: disable=no-member
                psi = np.asmatrix(u_u)*h_2*np.asmatrix(u_u)*np.transpose(psi_0)
                temp = np.asscalar(np.real(psi.getH()*m_m*psi))
                if temp > gap:
                    sample_total[n_1][n_2] = +1
                elif temp < -gap:
                    sample_total[n_1][n_2] = -1
                else:
                    sample_total[n_1][n_2] = 0

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        t_r = 0
        while t_r < (training_size+test_size):
            draw1 = np.random.choice(count)
            draw2 = np.random.choice(count)
            if sample_total[draw1][draw2] == +1:
                sample_a[t_r] = [2*np.pi*draw1/count, 2*np.pi*draw2/count]
                t_r += 1

        t_r = 0
        while t_r < (training_size+test_size):
            draw1 = np.random.choice(count)
            draw2 = np.random.choice(count)
            if sample_total[draw1][draw2] == -1:
                sample_b[t_r] = [2*np.pi*draw1/count, 2*np.pi*draw2/count]
                t_r += 1

        sample_train = [sample_a, sample_b]

        for lindex in range(training_size+test_size):
            label_train[lindex] = 0
        for lindex in range(training_size+test_size):
            label_train[training_size+test_size+lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2*(training_size+test_size), n))
        training_input = {key: (sample_train[label_train == k, :])[:training_size]
                          for k, key in enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[training_size:(
            training_size+test_size)] for k, key in enumerate(class_labels)}

        if plot_data:
            if not HAS_MATPLOTLIB:
                raise NameError('Matplotlib not installed. Plase install it before plotting')

            plt.show()
            fig2 = plt.figure()
            for k in range(0, 2):
                plt.scatter(sample_train[label_train == k, 0][:training_size],
                            sample_train[label_train == k, 1][:training_size])

            plt.title("Ad-hoc Data")
            plt.show()

    elif n == 3:
        for n_1 in range(count):
            for n_2 in range(count):
                for n_3 in range(count):
                    x_1 = steps*n_1
                    x_2 = steps*n_2
                    x_3 = steps*n_3
                    phi = x_1*np.kron(np.kron(z_m, j_m), j_m) + \
                        x_2*np.kron(np.kron(j_m, z_m), j_m) + \
                        x_3*np.kron(np.kron(j_m, j_m), z_m) + \
                        (np.pi-x_1)*(np.pi-x_2)*np.kron(np.kron(z_m, z_m), j_m) + \
                        (np.pi-x_2)*(np.pi-x_3)*np.kron(np.kron(j_m, z_m), z_m) + \
                        (np.pi-x_1)*(np.pi-x_3)*np.kron(np.kron(z_m, j_m), z_m)
                    u_u = scipy.linalg.expm(1j*phi)  # pylint: disable=no-member
                    psi = np.asmatrix(u_u)*h_3*np.asmatrix(u_u)*np.transpose(psi_0)
                    temp = np.asscalar(np.real(psi.getH()*m_m*psi))
                    if temp > gap:
                        sample_total[n_1][n_2][n_3] = +1
                        sample_total_a.append([n_1, n_2, n_3])
                    elif temp < -gap:
                        sample_total[n_1][n_2][n_3] = -1
                        sample_total_b.append([n_1, n_2, n_3])
                    else:
                        sample_total[n_1][n_2][n_3] = 0
                        sample_total_void.append([n_1, n_2, n_3])

        # Now sample randomly from sample_Total a number of times training_size+testing_size
        t_r = 0
        while t_r < (training_size+test_size):
            draw1 = np.random.choice(count)
            draw2 = np.random.choice(count)
            draw3 = np.random.choice(count)
            if sample_total[draw1][draw2][draw3] == +1:
                sample_a[t_r] = [2*np.pi*draw1/count, 2*np.pi*draw2/count, 2*np.pi*draw3/count]
                t_r += 1

        t_r = 0
        while t_r < (training_size+test_size):
            draw1 = np.random.choice(count)
            draw2 = np.random.choice(count)
            draw3 = np.random.choice(count)
            if sample_total[draw1][draw2][draw3] == -1:
                sample_b[t_r] = [2*np.pi*draw1/count, 2*np.pi*draw2/count, 2*np.pi*draw3/count]
                t_r += 1

        sample_train = [sample_a, sample_b]

        for lindex in range(training_size+test_size):
            label_train[lindex] = 0
        for lindex in range(training_size+test_size):
            label_train[training_size+test_size+lindex] = 1
        label_train = label_train.astype(int)
        sample_train = np.reshape(sample_train, (2*(training_size+test_size), n))
        training_input = {key: (sample_train[label_train == k, :])[:training_size]
                          for k, key in enumerate(class_labels)}
        test_input = {key: (sample_train[label_train == k, :])[training_size:(
            training_size+test_size)] for k, key in enumerate(class_labels)}

        if plot_data:
            if not HAS_MATPLOTLIB:
                raise NameError('Matplotlib not installed. Plase install it before plotting')
            sample_total_a = np.asarray(sample_total_a)
            sample_total_b = np.asarray(sample_total_b)
            x_1 = sample_total_a[:, 0]
            y_1 = sample_total_a[:, 1]
            z_1 = sample_total_a[:, 2]

            x_2 = sample_total_b[:, 0]
            y_2 = sample_total_b[:, 1]
            z_2 = sample_total_b[:, 2]

            fig1 = plt.figure()
            ax_1 = fig1.add_subplot(1, 1, 1, projection='3d')
            ax_1.scatter(x_1, y_1, z_1, c='#8A360F')
            plt.show()

            fig2 = plt.figure()
            ax_2 = fig2.add_subplot(1, 1, 1, projection='3d')
            ax_2.scatter(x_2, y_2, z_2, c='#683FC8')
            plt.show()

            sample_training_a = training_input['A']
            sample_training_b = training_input['B']

            x_1 = sample_training_a[:, 0]
            y_1 = sample_training_a[:, 1]
            z_1 = sample_training_a[:, 2]

            x_2 = sample_training_b[:, 0]
            y_2 = sample_training_b[:, 1]
            z_2 = sample_training_b[:, 2]

            fig1 = plt.figure()
            ax_1 = fig1.add_subplot(1, 1, 1, projection='3d')
            ax_1.scatter(x_1, y_1, z_1, c='#8A360F')
            ax_1.scatter(x_2, y_2, z_2, c='#683FC8')
            plt.show()

    return sample_total, training_input, test_input, class_labels


def sample_ad_hoc_data(sample_total, test_size, n):
    """ returns sample ad hoc data """

    class_labels = [r'A', r'B']  # copied from ad_hoc_data()
    count = 0
    if n == 2:
        count = 100
    elif n == 3:
        count = 20

    label_train = np.zeros(2*test_size)
    sample_a = [[0 for x in range(n)] for y in range(test_size)]
    sample_b = [[0 for x in range(n)] for y in range(test_size)]
    t_r = 0
    while t_r < (test_size):
        draw1 = np.random.choice(count)
        draw2 = np.random.choice(count)
        if sample_total[draw1][draw2] == +1:
            sample_a[t_r] = [2*np.pi*draw1/count, 2*np.pi*draw2/count]
            t_r += 1

    t_r = 0
    while t_r < (test_size):
        draw1 = np.random.choice(count)
        draw2 = np.random.choice(count)
        if sample_total[draw1][draw2] == -1:
            sample_b[t_r] = [2*np.pi*draw1/count, 2*np.pi*draw2/count]
            t_r += 1
    sample_train = [sample_a, sample_b]
    for lindex in range(test_size):
        label_train[lindex] = 0
    for lindex in range(test_size):
        label_train[test_size+lindex] = 1
    label_train = label_train.astype(int)
    sample_train = np.reshape(sample_train, (2 * test_size, n))
    test_input = {key: (sample_train[label_train == k, :])[:] for k, key in enumerate(class_labels)}
    return test_input
