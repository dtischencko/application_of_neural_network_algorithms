import numpy as np


class Neuron:

    def __init__(self, id_ne):
        self.last_indu = None
        self.id_ne = id_ne

    def set_indu_method(self, xs, ws, iter_num):
        self.last_indu = ws.iloc[self.id_ne].T.dot(xs.iloc[iter_num])
