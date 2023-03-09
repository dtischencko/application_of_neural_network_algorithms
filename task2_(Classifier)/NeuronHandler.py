def derivative(s):
    if s > 0:
        return 1
    else:
        return 0


class Neuron:

    def __init__(self, id_ne):
        self.__id_ne = id_ne
        self.last_indu = 0

    def __set_indu_field(self, xs, ws, iter_num):
        self.last_indu = ws.iloc[self.__id_ne].T.dot(xs.iloc[iter_num])

    def relu(self, xs, ws, iter_num):
        self.__set_indu_field(xs, ws, iter_num)  # find induction field
        if self.last_indu > 0:  # relu
            return self.last_indu
        else:
            return 0
