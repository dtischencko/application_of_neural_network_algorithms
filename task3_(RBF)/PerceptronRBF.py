import math

import numpy as np
import pandas as pd
import time


class Percept:

    LEARNING_RATE = 0.1

    def __init__(self, cs):
        self.__X = None
        self.__y = None
        self.full_func_signal = None
        self.core_g = 0
        self.mse_score = 0
        self.__G = []
        self.__cs = cs  # centers
        self.__hidden_neurons = [Percept.Neuron(amount_of_prev=1, num_of_neuron=i, center=self.__cs[i]) for i in range(len(cs))]
        self.__out_neuron = Percept.Neuron(amount_of_prev=len(cs), num_of_neuron=0, hidden=False)

    def transform_fit(self, X, y, is_singular=False):
        self.__G = []
        self.__X = X
        self.__y = y
        samples = [x for x in self.__X]
        for x in samples:  # rollup basis kernels
            self.__forward_propagation(x)
        self.core_g = self.__normal_equal(is_singular)

        errors = [math.sqrt(math.pow(self.predict(x_) - y_, 2)) for x_, y_ in zip(self.__X, self.__y)]
        self.mse_score = np.array(errors).sum() / len(self.__X)

    def predict(self, x):
        self.__forward_propagation(x, is_predict=True)
        return self.__out_neuron.estimate_kernels(self.__out_neuron.func_signal)

    def __forward_propagation(self, x, is_predict=False):
        for h_neuron in self.__hidden_neurons:  # h_neurons
            h_neuron.find_kernel_gaussian_RBF(x)

        self.full_func_signal = np.array([h_neuron.func_signal for h_neuron in self.__hidden_neurons])
        if not is_predict:
            self.__G.append(self.full_func_signal)
        else:
            self.__out_neuron.func_signal = self.full_func_signal

        # print(self.__G)

    def __normal_equal(self, is_singular=False):
        self.__G = np.array(self.__G).reshape((len(self.__X), len(self.__cs)))
        if not is_singular:
            core_g = np.linalg.inv(self.__G.T @ self.__G) @ self.__G.T
            theta = core_g @ self.__y
        else:
            core_g = np.linalg.pinv(self.__G)
            theta = core_g @ self.__y
        print(core_g, "\n")
        self.__out_neuron.set_full_ws(theta)
        return core_g

    class Neuron:

        def __init__(self, amount_of_prev, num_of_neuron, center=None, hidden=True):  # num_of_neuron for identify out neuron
            self.__num_of_neuron = None
            self.__center = None
            self.func_signal = None
            self.__isHidden = hidden
            self.__num_of_prev = amount_of_prev
            if not self.is_hidden():
                self.__ws = pd.Series([0 for i in range(amount_of_prev)])  # weights
            else:
                self.__ws = None
                self.__center = center
            self.__num_of_neuron = num_of_neuron

        def find_kernel_gaussian_RBF(self, x):
            self.func_signal = math.exp(-0.5*math.pow(x - self.__center, 2))  # RBF kernel

        def estimate_kernels(self, full_func_signal):
            return full_func_signal @ self.get_full_ws().T

        def get_ws(self, i):
            return self.__ws[i]

        def set_ws(self, i, value):
            self.__ws.iloc[i] = value

        def get_full_ws(self):
            return self.__ws

        def set_full_ws(self, values):
            self.__ws = values

        def is_hidden(self):
            return self.__isHidden


# X = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
# y = [-0.48, -0.78, -0.83, -0.67, -0.20, 0.70, 1.48, 1.17, 0.20]
# cs = [-2.0, -1.0, 0.0, 1.0, 2.0]
#
# p = Percept(cs=cs)
# p.transform_fit(X, y)