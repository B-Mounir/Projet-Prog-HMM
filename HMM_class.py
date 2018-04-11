import numpy as np
import random


class HMM:
    """Define an HMM"""

    def __init__(self, nbl, nbs, initial, transitions, emissions):
        # The number of letters
        self.nbl = nbl
        # The number of states
        self.nbs = nbs
        # The vector defining the initial weights
        self.initial = initial
        # The array defining the transitions
        self.transitions = transitions
        # The list of vectors defining the emissions
        self.emissions = emissions

    @property
    def nbl(self):
        return self.__nbl

    @nbl.setter
    def nbl(self, x):
        """Modify the number of letters"""
        if not isinstance(x, int):
            raise ValueError("Value Error : should be an integer")
        elif x <= 0:
            raise ValueError("Value Error : should be positive")
        self.__nbl = x

    @property
    def nbs(self):
        return self.__nbs

    @nbs.setter
    def nbs(self, x):
        """Modify the numbers of states"""
        if not isinstance(x, int):
            raise ValueError("Value Error : please enter an integer")
        elif x <= 0:
            raise ValueError("Value Error : should be superior to 0")
        self.__nbs = x

    @property
    def initial(self):
        return self.__initial

    @initial.setter
    def initial(self, x):
        """Modify the initial transition"""
        if not isinstance(x, np.ndarray):
            raise ValueError("Value Error : should be an array")
        elif x.shape != (1, self.nbs):
            raise ValueError("Value Error : shape should be (1, nbs)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array should be float")
        elif not np.isclose(x.sum(), [1.0], 0.001):
            raise ValueError("Value Error : sum of initial probabilities should be equal to 1")
        self.__initial = x

    @property
    def transitions(self):
        return self.__transitions

    @transitions.setter
    def transitions(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError("Value Error : should be an array")
        elif x.shape != (self.nbs, self.nbs):
            raise ValueError("Value Error : shape should be (nbs, nbs)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array should be float")
        for y in x.sum(axis=1):
            if not np.isclose(y, [1.0], 0.001):
                raise ValueError("Value Error : sum of transitions' probabilities from each state should be 1")
        self.__transitions = x

    @property
    def emissions(self):
        return self.__emissions

    @emissions.setter
    def emissions(self, x):
        """Modify the emission vector"""
        if not isinstance(x, np.ndarray):
            raise ValueError("Value Error : should be an array")
        elif x.shape != (self.nbs, self.nbl):
            raise ValueError("Value Error : shape should be (nbs, nbl)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array should be float")
        for y in x.sum(axis=1):
            if not np.isclose(y, [1.0], 0.001):
                raise ValueError("Value Error : sum of transitions' probabilities from each state should be 1")

    @staticmethod
    def draw_multinomial(l):
        h = np.random.multinomial(1, l)
        for i in range(len(h)):
            if h[i] == 1:
                return i

    def gen_rand(self, n):
        i = HMM.draw_multinomial(self.initial[0])
        m = np.zeros((1, n))
        for j in range(n):
            m[0][j] = HMM.draw_multinomial(self.emissions[i])
            i = HMM.draw_multinomial(self.transitions[i])
        return m


a = HMM(2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.5, 0.5],[0.7, 0.3]]))
print(a.gen_rand(10))