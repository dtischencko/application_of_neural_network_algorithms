import numpy as np


class Neuron:

    last_ind = np.NaN
    id_ne = np.NaN

    def __init__(self, id_ne):
        self.id_ne = id_ne

    def set_indu_method(self, xs, ws, iter_num):
        self.last_ind = ws.iloc[self.id_ne].T.dot(xs.iloc[iter_num])


