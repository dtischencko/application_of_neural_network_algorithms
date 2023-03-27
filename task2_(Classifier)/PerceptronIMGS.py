import math
import numpy as np
import pandas as pd
import time


class PerceptIMGS:

    LEARNING_RATE = 0.1

    def __init__(self, X, y, amount_of_hidden_neurons):
        self.__X = X
        self.__y = y
        self.__amount_of_neurons = amount_of_hidden_neurons
        self.__hidden_neurons = [PerceptIMGS.Neuron(amount_of_prev=9, num_of_neuron=i) for i in range(self.__amount_of_neurons)]
        self.__out_neurons = [PerceptIMGS.Neuron(amount_of_prev=self.__amount_of_neurons, num_of_neuron=i, hidden=False) for i in range(4)]

    def transform_fit(self, epochs=1):
        start_time = time.time()
        samples = [(x, label) for x, label in zip(self.__X.values, self.__y.values)]
        # print(self.__X.values, self.__y.values)
        for epoch in range(epochs):
            print("Epoch#", epoch+1, "/", epochs)
            np.random.shuffle(samples)
            for (x, label) in samples:
                # print("Signal:", x, "\nLabel:", label)
                hypothesis = self.__forward_propagation(x)
                print(hypothesis, "\n")
                self.__backward_propagation(hypothesis, label, x)

        print((time.time() - start_time) % 60, "sec")

    def predict(self, x):
        hypothesis = self.__forward_propagation(x)
        return pd.Series(hypothesis)

    def __forward_propagation(self, x):
        for h_neuron in self.__hidden_neurons:  # h_neurons
            h_neuron.find_func_signal(x)

        full_func_signal = [h_neuron.func_signal for h_neuron in self.__hidden_neurons]
        for o_neuron in self.__out_neurons:
            o_neuron.find_func_signal(full_func_signal)

        return [o_neuron.func_signal for o_neuron in self.__out_neurons]  # returning hypothesis

    def __backward_propagation(self, hypothesis, label, x):
            for neuron in self.__out_neurons:  #  find all local gradients
                l = label[neuron.get_num_of_neuron()]
                # print("FOR ", l)
                neuron.find_local_gradient(self.__out_neurons, hypothesis=hypothesis, label=l)
            for neuron in self.__hidden_neurons:
                neuron.find_local_gradient(self.__out_neurons)

            #  do correlation all weights
            self.__correlate_algorithm(self.__out_neurons, self.__hidden_neurons)  # correlate out_neurons
            self.__correlate_algorithm(self.__hidden_neurons, x)  # correlate hidden_neurons

    def __correlate_algorithm(self, neurons, prev_neurons = None):
        for neuron in neurons:
            weights = neuron.get_full_ws()
            for i in range(len(weights)):
                w = weights[i]
                prev_neuron = prev_neurons[i]
                neuron.set_ws(i, self.__correlate_weight(neuron, prev_neuron, w))


    def __correlate_weight(self, neuron, prev_neuron, weight):
        if neuron.is_hidden():  # when hidden neuron --> functional signal from previous is LAST_INDU_FIELD
            delta_w = PerceptIMGS.LEARNING_RATE * neuron.local_gradient * prev_neuron  # for hidden neuron --> previous neuron is X signal
        else:  # when out neurons --> functional signal from hidden neuron
            delta_w = PerceptIMGS.LEARNING_RATE * neuron.local_gradient * prev_neuron.func_signal
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

        def find_func_signal(self, x):  # with sigmoid logical
            self.last_indu_field = self.__ws @ x
            self.func_signal = self.__sigmoid(self.last_indu_field)

        def find_local_gradient(self, out_neurons, hypothesis=None, label=None):
            if not self.__isHidden:
                error = label - hypothesis[self.__num_of_neuron]
                self.local_gradient = error * self.__derivative(self.last_indu_field)
            else:
                error = 0
                for neuron in out_neurons:  # for all out neurons ==> iter_neuron local gradient * (this_neuron --w--> prev_neuron)
                    error += neuron.local_gradient * neuron.get_ws(i=self.__num_of_neuron)
                self.local_gradient = error * self.__derivative(self.last_indu_field)

        def __sigmoid(self, indu_field):
            return 1 / (1 + math.exp(-indu_field))

        def __derivative(self, indu_field):
            return self.__sigmoid(indu_field) * (1 - self.__sigmoid(indu_field))

        def get_ws(self, i):
            return self.__ws[i]

        def set_ws(self, i, value):
            self.__ws.iloc[i] = value

        def get_full_ws(self):
            return self.__ws

        def get_num_of_neuron(self):
            return self.__num_of_neuron

        def is_hidden(self):
            return self.__isHidden


# x1 = np.array([1,0,1,0,1,0,1,0,1])  # X
# x2 = np.array([1,0,1,0,1,0,0,1,0])  # Y
# x3 = np.array([0,1,0,0,1,0,0,1,0])  # I
# x4 = np.array([1,0,0,1,0,0,1,1,1])  # L
# y1 = np.array([1,0,0,0])
# y2 = np.array([0,1,0,0])
# y3 = np.array([0,0,1,0])
# y4 = np.array([0,0,0,1])
# X = pd.DataFrame([x1, x2, x3, x4])
# y = pd.DataFrame([y1, y2, y3, y4])
# p = PerceptIMGS(X, y, amount_of_hidden_neurons=10)
