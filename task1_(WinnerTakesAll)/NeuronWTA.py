import numpy as np
import math


class Neuron:

    THRESHOLD = 10

    def __init__(self, id_ne):
        self.last_indu = None
        self.id_ne = id_ne
        self.__count_learn = 0  # after 10 iter going to delay
        self.__delay = (-1) * self.THRESHOLD
        self.canLearning = True

    def set_indu_method(self, xs, ws, iter_num, is_sigmoid=False):
        if is_sigmoid:
            self.last_indu = math.tanh(ws.iloc[self.id_ne].T.dot(xs.iloc[iter_num]))
            return
        self.last_indu = ws.iloc[self.id_ne].T.dot(xs.iloc[iter_num])

    def add_count_learn(self):
        self.__count_learn += 1
        if self.__count_learn > self.THRESHOLD:
            self.__delay = (-1) * self.THRESHOLD
            self.__count_learn = 0
            self.canLearning = False

    def take_delay(self):
        self.__delay += 1
        if self.__delay == 0:
            self.canLearning = True
