import numpy as np


class Task(object):
    def __init__(self, dim, fnc, ub, lb, dqltask):
        self.dim = dim
        self.fnc = fnc
        self.lb = lb
        self.ub = ub
        self.dqltask = dqltask

    def decode(self, rnvec):
        # nvars = rnvec[:self.dim]
        return self.lb + rnvec * (self.ub - self.lb)

    def encode(self, vec):
        return (vec - self.lb)/(self.ub - self.lb)
