#################################################################################
# Title : HMM_class.py                                                          #
# Autors : AMRAM Yassine, BAZELAIRE Guillaume, BENNADJI Mounir, WEBERT Vincent  #
# Date : 18-05-07                                                               #
#################################################################################

import numpy as np
import random
import copy
import math


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

    def __repr__(self):
        return self.nbl, self.nbs, self.initial, self.transitions, self.emissions

    def __str__(self):
        return str(self.__repr__())

    @property
    def nbl(self):
        return self.__nbl

    @nbl.setter
    def nbl(self, x):
        """Modify the number of letters"""
        if not isinstance(x, int):
            raise TypeError("Value Error : should be an integer")
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
            raise TypeError("Value Error : should be an integer")
        elif x <= 0:
            raise ValueError("Value Error : should be superior to 0")
        self.__nbs = x

    @property
    def initial(self):
        return self.__initial

    @initial.setter
    def initial(self, x):
        """Modify the vector defining the initial weights"""
        if not isinstance(x, np.ndarray):
            raise TypeError("Value Error : should be an array")
        elif x.shape != (1, self.nbs):
            raise ValueError("Value Error : shape should be (1, nbs)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array should be float")
        elif not np.isclose(x.sum(), [1.0], 0.01):
            raise ValueError("Value Error : sum of initial probabilities should be equal to 1")
        self.__initial = x

    @property
    def transitions(self):
        return self.__transitions

    @transitions.setter
    def transitions(self, x):
        """Modify the array defining the transitions weights"""
        if not isinstance(x, np.ndarray):
            raise TypeError("Value Error : should be an array")
        elif x.shape != (self.nbs, self.nbs):
            raise ValueError("Value Error : shape should be (nbs, nbs)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array should be float")
        for y in x.sum(axis=1):
            if not np.isclose(y, [1.0], 0.01):
                raise ValueError("Value Error : sum of transitions' probabilities from each state should be 1")
        self.__transitions = x

    @property
    def emissions(self):
        return self.__emissions

    @emissions.setter
    def emissions(self, x):
        """Modify the array defining the emissions weights"""
        if not isinstance(x, np.ndarray):
            raise TypeError("Value Error : should be an array")
        elif x.shape != (self.nbs, self.nbl):
            raise ValueError("Value Error : shape should be (nbs, nbl)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array should be float")
        for y in x.sum(axis=1):
            if not np.isclose(y, [1.0], 0.01):
                raise ValueError("Value Error : sum of transitions' probabilities from each state should be 1")
        self.__emissions = x

    @staticmethod
    def load(adr):
        """load a text file and return the HMM corresponding"""
        with open(adr, 'r') as file:
            lines = file.readlines()
            nbl = int(lines[1])
            nbs = int(lines[3])
            initial = []
            i = 5
            while lines[i][0] != '#':
                initial.append(float(lines[i]))
                i += 1
            i += 1
            initial = np.array([initial])
            transitions = []
            while lines[i][0] != '#':
                transitions.append([float(e) for e in lines[i].split()])
                i += 1
            i += 1
            transitions = np.array(transitions)
            emissions = []
            while i < len(lines):
                emissions.append([float(e) for e in lines[i].split()])
                i += 1
            emissions = np.array(emissions)
        return HMM(nbl, nbs, initial, transitions, emissions)

    def save(self, adr):
        """save a HMM in a text file"""
        with open(adr, 'w') as HLM:
            HLM.write('# The number of letters')
            HLM.write('\n' + str(self.nbl))
            HLM.write('\n' + '# The number of states')
            HLM.write('\n' + str(self.nbs))
            HLM.write('\n' + '# The initial transitions')
            for e in self.initial[0]:
                HLM.write('\n' + str(e))
            HLM.write('\n' + '# The internal transitions')
            for l in self.transitions:
                HLM.write('\n')
                for c in l:
                    HLM.write(str(c) + ' ')
            HLM.write('\n' + '# The emissions')
            for l in self.emissions:
                HLM.write('\n')
                for c in l:
                    HLM.write(str(c) + ' ')

    @staticmethod
    def draw_multinomial(l):
        """
        :param l: array of probabilities (float) whose sum is equal to 1
        :return: the index coresponding to the result of a draw which respects the multinomial model defined by l
        """
        if not isinstance(l, np.ndarray):
            raise TypeError("Value Error : should be an array")
        elif not np.isclose(l.sum(), [1.0], 0.01):
            raise ValueError("Value Error : sum of probabilities should be equal to 1")
        elif l.ndim != 1:
            raise ValueError("Value Error : should be a 1 dimension array")

        m = []
        s = 0
        for i in range(len(l)):
            m += [s]
            s += l[i]
        m += [s]
        x = random.random()
        for i in range(len(m)):
            if m[i] <= x <= m[i+1]:
                return i

    def gen_rand(self, n):
        """
        :param n: Integer
        :return: a random sequence with a length equal to n corresponding to a HMM
        """
        if not isinstance(n, int):
            raise TypeError("Value Error : should be an integer")

        i = HMM.draw_multinomial(self.initial[0])
        m = []
        for j in range(n):
            m.append(HMM.draw_multinomial(self.emissions[i]))
            i = HMM.draw_multinomial(self.transitions[i])
        return m

    def pfw(self, w):
        """
        :param w: Sequence of observable states
        :return: the probability of the sequence w with a particular HMM using forward
        """
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        n = len(w)
        f = np.zeros((self.nbs,))
        for k in range(self.nbs):
            f[k] = self.initial[0][k] * self.emissions[k][w[0]]
        for i in range(1, n):
            f = np.dot(f, self.transitions) * self.emissions[:, w[i]]
        return f.sum()

    def pbw(self, w):
        """
        :param w: Sequence of observable states
        :return: the probability of the sequence w with a particular HMM using backward
        """
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        n = len(w)
        b = np.array([1]*self.nbs)
        for i in range(n - 1, 0, -1):
            b = np.dot(self.transitions, self.emissions[:, w[i]] * b)
        p = b * self.initial * self.emissions[:, w[0]]
        return p.sum()

    def predit(self, w):
        """
        :param w: Sequence of observable states
        :return: predict the next letter of the sequence w using a particular HMM
        """
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        H = self.initial[0]
        for i in range(len(w)):
            H = (self.transitions.T * self.emissions[:, w[i]].T) @ H
        P = []
        for l in range(self.nbl):
            P += [self.emissions[:, l] @ H]
        return P.index(max(P))

    def Vraisemblance(self, S):
        """
       :param S: list of observable states sequences
       :return: the likelihood of the list of sequences S
       """
        if not isinstance(S, list):
            raise TypeError("Value Error : should be a list")
        for w in S:
            if not isinstance(w, list):
                raise TypeError("Value Error : should be a list")

        res = 1
        for w in S:
            res = res * self.pfw(w)
        return res

    def logV(self, S):
        """
        :param S: list of observable states sequences
        :return: the log likelihood of the list of sequences S
        """
        if not isinstance(S, list):
            raise TypeError("Value Error : should be a list")
        for w in S:
            if not isinstance(w, list):
                raise TypeError("Value Error : should be a list")

        somme = 0
        for w in S:
            somme += np.log(self.pfw(w))
        return somme

    def viterbi(self, w):
        """
        :param w: Sequence of observable states
        :return: The Viterbi path of w ans its log probability
        """
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        n = len(w)
        c_1 = []
        c_2 = []
        liste_etats = []
        p_1 = self.initial[0] * self.emissions[:, w[0]]
        p_2 = self.initial[0] * self.emissions[:, w[0]]
        for i in range(self.nbs):
            c_1 += [[i]]
            c_2 += [[i]]
            liste_etats += [i]
        for i in range(1, n):
            for k in range(self.nbs):
                m = 0
                l_max = 0
                for l in range(self.nbs):
                    a = m
                    b = p_1[l] * self.transitions[l, k]
                    m = max(a, b)
                    if m == b:
                        l_max = l
                c_2[k] = c_1[l_max] + [k]
                p_2[k] = m * self.emissions[k, w[i]]
            c_1 = copy.deepcopy(c_2)
            p_1 = copy.deepcopy(p_2)
        return c_2[np.argmax(p_2)], np.log(np.max(p_2))

    @staticmethod
    def gen_HMM(nbs, nbl):
        """
        :param nbs: Number of states
        :param nbl: Number of letters
        :return: A HMM randomly generated with nbs states ans nbl letters
        """
        if not isinstance(nbs, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(nbl, int):
            raise TypeError("Value Error : should be an integer")

        random.seed()
        sum = 0
        initial = []
        for i in range(nbs):
            x = random.random()
            initial += [x]
            sum += x
        for i in range(nbs):
            initial[i] /= sum
        transitions = []
        for j in range(nbs):
            transitions += [[]]
            sum = 0
            for i in range(nbs):
                x = random.random()
                transitions[j] += [x]
                sum += x
            for i in range(nbs):
                transitions[j][i] /= sum
        emissions = []
        for j in range(nbs):
            emissions += [[]]
            sum = 0
            for i in range(nbl):
                x = random.random()
                emissions[j] += [x]
                sum += x
            for i in range(nbl):
                emissions[j][i] /= sum
        initial = np.array([initial])
        transitions = np.array(transitions)
        emissions = np.array(emissions)
        return HMM(nbl, nbs, initial, transitions, emissions)

    # Baum-Welch :

    def f(self, w):
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        f = np.zeros((self.nbs, len(w)))
        f[:, 0] = self.initial[0] * self.emissions[:, w[0]]
        for i in range(1, len(w)):
            f[:, i] = np.dot(f[:, i - 1], self.transitions) * self.emissions[:, w[i]]
        return f

    def b(self, w):
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        b = np.zeros((self.nbs, len(w)))
        b[:, len(w) - 1] = np.array([1] * self.nbs)
        for i in range(len(w) - 2, -1, -1):
            b[:, i] = np.dot(self.transitions, self.emissions[:, w[i + 1]] * b[:, i + 1])
        return b

    def gamma(self, w):
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        f = self.f(w)
        b = self.b(w)
        return (f * b) / np.einsum('kt,kt->t', b, f)

    def xi(self, w):
        if not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        f = self.f(w)[:, :-1]
        b = self.b(w)[:, 1:]
        emissions = self.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, self.transitions, emissions, b)
        v = np.einsum('kt,kl,lt,lt->t', f, self.transitions, emissions, b)
        somme = np.tile(v, (self.nbs, self.nbs, 1))
        xi = xi / somme
        return xi

    @staticmethod
    def bw1(m0, S):
        """
        :param m0: HMM
        :param S: list of observable states sequences
        :return: the HMM updated using Baum-Welch algorithm
        """
        if not isinstance(S, list):
            raise TypeError("Value Error : should be a list")
        for w in S:
            if not isinstance(w, list):
                raise TypeError("Value Error : should be a list")
        if not isinstance(m0, HMM):
            raise TypeError("Value Error : should be a HMM")

        pi = np.zeros(m0.nbs)
        for j in range(len(S)):
            pi += np.array(m0.gamma(S[j])[:, 0])
        T = np.zeros((m0.nbs, m0.nbs))
        for j in range(len(S)):
            for t in range(len(S[j]) - 1):
                T += m0.xi(S[j])[:, :, t]
        O = np.zeros((m0.nbs, m0.nbl))
        for j in range(len(S)):
            gamma = m0.gamma(S[j])
            for t in range(len(S[j])):
                O[:, S[j][t]] += gamma[:, t]
        maj = HMM(m0.nbl, m0.nbs, np.array([pi / pi.sum()]), (T.T / T.sum(1)).T, (O.T / O.sum(1)).T)
        return maj

    @staticmethod
    def bw2(nbs, nbl, S, N):
        """
        :param nbs: Number of states (Integer)
        :param nbl: Number of letters (Integer)
        :param S: List of observable states sequences
        :param N: Integer
        :return: A HMM randomly generated with nbs states and nbl letters updated N times using bw1
                to increase the likelihood of S
        """
        if not isinstance(nbs, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(nbl, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(N, int):
            raise TypeError("Value Error : should be an integer")
        if not isinstance(S, list):
            raise TypeError("Value Error : should be a list")
        for w in S:
            if not isinstance(w, list):
                raise TypeError("Value Error : should be a list")

        M = HMM.gen_HMM(nbs, nbl)
        for i in range(N):
            M = HMM.bw1(M, S)
        return M

    @staticmethod
    def bw3(nbs, nbl, w, N, M):
        """
        :param nbs: Number of states (Integer)
        :param nbl: Number of letters (Integer)
        :param w: Sequence of observable states
        :param N: Integer
        :param M: Integer
        :return: The HMM Mi (0 <= i <= M-1) genrated with bw2 which maximize the likelihood of w
        """
        if not isinstance(nbs, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(nbl, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(N, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(M, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(w, list):
            raise TypeError("Value Error : should be a list")

        max_logV = -math.inf
        hmm = None
        for i in range(M):
            h = HMM.bw2(nbs, nbl, [w], N)
            logV = h.logV([w])
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm

    @staticmethod
    def bw4(nbs, nbl, S, N, M):
        """
        :param nbs: Number of states (Integer)
        :param nbl: Number of letters (Integer)
        :param S: List of observable states sequences
        :param N: Integer
        :param M: Integer
        :return: The HMM Mi (0 <= i <= M-1) generated with bw2 which maximize the likelihood of S
        """
        if not isinstance(nbs, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(nbl, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(N, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(M, int):
            raise TypeError("Value Error : should be an integer")
        if not isinstance(S, list):
            raise TypeError("Value Error : should be a list")
        for w in S:
            if not isinstance(w, list):
                raise TypeError("Value Error : should be a list")

        max_logV = -math.inf
        hmm = None
        for i in range(M):
            h = HMM.bw2(nbs, nbl, S, N)
            logV = h.logV(S)
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm

    @staticmethod
    def bw2_limite(nbs, nbl, S, limite, tolerance):
        """
        :param nbs: Number of states (Integer)
        :param nbl: Number of letters (Integer)
        :param S: List of observable states sequences
        :param limite: Number of iterations where the likelihood is stabilised (Integer)
        :param tolerance: Float
        :return: A HMM randomly generated with nbs states and nbl letters updated using bw1
                while the likelihood of S is not stabilised
        """
        if not isinstance(nbs, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(nbl, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(limite, int):
            raise TypeError("Value Error : should be an integer")
        if not isinstance(S, list):
            raise TypeError("Value Error : should be a list")
        for w in S:
            if not isinstance(w, list):
                raise TypeError("Value Error : should be a list")

        M = HMM.gen_HMM(nbs, nbl)
        l = limite
        log_vs = np.zeros((l,))
        i = 0
        logv = 10
        while not np.allclose(log_vs, logv, atol=tolerance):
            M = HMM.bw1(M, S)
            logv = M.logV(S)
            log_vs[i % l] = logv
            i = i + 1
        return M, i

    @staticmethod
    def bw4_limite(nbs, nbl, S, limite, tolerance, M):
        """
        :param nbs: Number of states (Integer)
        :param nbl: Number of letters (Integer)
        :param S: List of observable states sequences
        :param limite: Integer
        :param limite: Integer
        :param tolerance: Float
        :param M: Integer
        :return: The HMM Mi (0 <= i <= M-1) generated with bw2_limite which maximize the likelihood of S
        """
        if not isinstance(nbs, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(nbl, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(limite, int):
            raise TypeError("Value Error : should be an integer")
        elif not isinstance(M, int):
            raise TypeError("Value Error : should be an integer")
        if not isinstance(S, list):
            raise TypeError("Value Error : should be a list")
        for w in S:
            if not isinstance(w, list):
                raise TypeError("Value Error : should be a list")

        max_logV = -math.inf
        hmm = None
        for i in range(M):
            h = HMM.bw2_limite(nbs, nbl, S, limite, tolerance)[0]
            logV = h.logV(S)
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm
