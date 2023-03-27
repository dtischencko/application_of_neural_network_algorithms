import math
import numpy as np
import pandas as pd
import time


class Percept:

    LEARNING_RATE = 0.1

    def __init__(self, X, y, amount_of_hidden_neurons, is_ADLiNE=False):
        if is_ADLiNE:
            outs = 1
        else:
            outs = 2
        self.__X = X
        self.__y = y
        self.__y_estimating = []
        self.__amount_of_neurons = amount_of_hidden_neurons
        self.__hidden_neurons = [Percept.Neuron(amount_of_prev=2, num_of_neuron=i) for i in range(self.__amount_of_neurons)]
        self.__out_neurons = [Percept.Neuron(amount_of_prev=self.__amount_of_neurons, num_of_neuron=i, hidden=False) for i in range(outs)]  # 0neuron->True, 1neuron->False

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.is_ADLiNE = is_ADLiNE

    def __find_statistics(self):
        self.__y_estimating.clear()
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.precision = None
        self.recall = None
        self.f1_score = None

        for x in self.__X.values:
            self.__y_estimating.append(self.predict(x))

        l = np.where(self.__y, 1, 0)
        p = np.where(self.__y_estimating, 1, 0)
        l_plus_p = l+p
        l_minus_p = l-p
        print(l, p, "\n", l_plus_p, l_minus_p)
        for i in range(len(l_plus_p)):
            if l_plus_p[i] == 2:
                self.TP += 1
            elif l_plus_p[i] == 0:
                self.TN += 1
            elif l_minus_p[i] == -1:
                self.FP += 1
            elif l_minus_p[i] == 1:
                self.FN += 1

        print(self.TP, self.TN, self.FP, self.FN)


        divider = (self.TP + self.FP)
        if divider == 0:
            divider = 1
        self.precision = self.TP / divider

        divider = (self.TP + self.FN)
        if divider == 0:
            divider = 1
        self.recall = self.TP / divider

        divider = (self.precision + self.recall)
        if divider == 0:
            divider = 1
        self.f1_score = 2 * self.precision * self.recall / divider

    def transform_fit(self, epochs=1):
        start_time = time.time()
        samples = [(x, label) for x, label in zip(self.__X.values, self.__y)]
        for epoch in range(epochs):
            print("Epoch#", epoch+1, "/", epochs)
            np.random.shuffle(samples)
            for (x, label) in samples:
                hypothesis = self.__forward_propagation(x)
                print(hypothesis)
                self.__backward_propagation(hypothesis, label, x)

        self.__find_statistics()
        print((time.time() - start_time) % 60, "sec")

    def predict(self, x):
        hypothesis = self.__forward_propagation(x)
        if not self.is_ADLiNE:
            return bool(pd.Series(hypothesis).idxmax())
        if self.is_ADLiNE:
            if hypothesis[0] == -1:
                return False
            elif hypothesis[0] == 1:
                return True


    def __forward_propagation(self, x):
        for h_neuron in self.__hidden_neurons:  # h_neurons
            h_neuron.find_func_signal(x, self.is_ADLiNE)

        full_func_signal = [h_neuron.func_signal for h_neuron in self.__hidden_neurons]
        for o_neuron in self.__out_neurons:
            o_neuron.find_func_signal(full_func_signal, self.is_ADLiNE)

        return [o_neuron.func_signal for o_neuron in self.__out_neurons]  # returning hypothesis

    def __backward_propagation(self, hypothesis, label, x):
        # if not self.is_ADLiNE:  #  find all local gradients
            for neuron in self.__out_neurons:
                neuron.find_local_gradient(self.__out_neurons, is_ADLiNE=self.is_ADLiNE, hypothesis=hypothesis, label=label)
            for neuron in self.__hidden_neurons:
                neuron.find_local_gradient(self.__out_neurons, is_ADLiNE=self.is_ADLiNE)

            #  do correlation all weights
            self.__correlate_algorithm(self.__out_neurons, self.__hidden_neurons)  # correlate out_neurons
            self.__correlate_algorithm(self.__hidden_neurons, x)  # correlate hidden_neurons
        # else:

    def __correlate_algorithm(self, neurons, prev_neurons = None):
        for neuron in neurons:
            weights = neuron.get_full_ws()
            for i in range(len(weights)):
                w = weights[i]
                prev_neuron = prev_neurons[i]
                neuron.set_ws(i, self.__correlate_weight(neuron, prev_neuron, w))


    def __correlate_weight(self, neuron, prev_neuron, weight):
        if neuron.is_hidden():  # when hidden neuron --> functional signal from previous is LAST_INDU_FIELD
            delta_w = Percept.LEARNING_RATE * neuron.local_gradient * prev_neuron  # for hidden neuron --> previous neuron is X signal
        else:  # when out neurons --> functional signal from hidden neuron
            delta_w = Percept.LEARNING_RATE * neuron.local_gradient * prev_neuron.func_signal
        return weight + delta_w

    class Neuron:

        def __init__(self, amount_of_prev, num_of_neuron, hidden=True):  # num_of_neuron for identify out neuron
            self.__num_of_neuron = None
            self.local_gradient = None
            self.func_signal = None
            self.last_indu_field = None
            self.__isHidden = hidden
            self.__num_of_prev = amount_of_prev
            self.__ws = pd.Series([np.random.randn() for i in range(amount_of_prev)])  # weights
            self.__num_of_neuron = num_of_neuron

        def find_func_signal(self, x, is_ADLiNE):  # with sigmoid logical
            self.last_indu_field = self.__ws @ x
            if not is_ADLiNE:
                self.func_signal = self.__sigmoid(self.last_indu_field)
            else:
                self.func_signal = self.__signum(self.last_indu_field)

        def find_local_gradient(self, out_neurons, is_ADLiNE, hypothesis=None, label=None):
            if not self.__isHidden:
                error = int(label) - hypothesis[self.__num_of_neuron]
                if not is_ADLiNE:
                    self.local_gradient = error * self.__derivative(self.last_indu_field)
                else:
                    self.local_gradient = error * self.__derivative_of_signum(self.last_indu_field)
            else:
                error = 0
                for neuron in out_neurons:  # for all out neurons ==> iter_neuron local gradient * (this_neuron --w--> prev_neuron)
                    error += neuron.local_gradient * neuron.get_ws(i=self.__num_of_neuron)
                if not is_ADLiNE:
                    self.local_gradient = error * self.__derivative(self.last_indu_field)
                else:
                    self.local_gradient = error * self.__derivative_of_signum(self.last_indu_field)
        def __sigmoid(self, indu_field):
            return 1 / (1 + math.exp(-indu_field))

        def __signum(self, indu_field):
            if indu_field > 0:
                return 1
            else:
                return -1

        def __derivative(self, indu_field):
            return self.__sigmoid(indu_field) * (1 - self.__sigmoid(indu_field))

        def __derivative_of_signum(self, indu_field):
            return indu_field * self.__signum(indu_field)

        def get_ws(self, i):
            return self.__ws[i]

        def set_ws(self, i, value):
            self.__ws.iloc[i] = value

        def get_full_ws(self):
            return self.__ws

        def is_hidden(self):
            return self.__isHidden

#
# x1 = np.array([np.random.uniform(0, 1) for i in range(20)])
# x2 = np.array([np.random.uniform(0, 1) for i in range(20)])
# y = np.array(x1 > x2)
# X = pd.DataFrame([x1, x2]).T
# p = Percept(X, y, amount_of_hidden_neurons=10, is_ADLiNE=True)
