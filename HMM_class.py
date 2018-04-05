import numpy as np 
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
            raise ValueError("Value Error : please enter an integer")
        elif x <= 0:
            raise ValueError("Value Error : should be superior to 0")
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
            raise ValueError("Value Error : please enter an array")
        elif x.shape != (1, self.nbs):
            raise ValueError("Value Error : wrong shape, should be (1, nbs)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array with wrong type, should be float")
        elif not np.isclose(x.sum(), [1.0], 0.001):
            raise ValueError("Value Error : sum of initial probabilities should be equal to 1")

        self.__initial = x

    @property
    def transitions(self):
        return self.__transitions

    @transitions.setter
    def transitions(self, x):
        if not isinstance(x, np.ndarray):
            raise ValueError("Value Error : please enter an array")
        elif x.shape != (self.nbs, self.nbs):
            raise ValueError("Value Error : wrong shape, should be (nbs, nbs)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array with wrong type, should be float")
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
            raise ValueError("Value Error : please enter an array")
        elif x.shape != (self.nbs, self.nbl):
            raise ValueError("Value Error : wrong shape, should be (nbs, nbl)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array with wrong type, should be float")
        for y in x.sum(axis=1):
            if not np.isclose(y, [1.0], 0.001):
                raise ValueError("Value Error : sum of transitions' probabilities from each state should be 1")

        self.__emissions = x

    @staticmethod
    def draw_multinomial(l):
        h = np.random.multinomial(1, l)
        for i in range(len(h)):
            if h[i] == 1:
                return i
