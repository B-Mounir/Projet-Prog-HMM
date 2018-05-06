import numpy as np
import random
import time
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
            raise TypeError("Value Error : please enter an integer")
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
        with open(adr, 'r') as HLM:
            lines = HLM.readlines()
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
        """return the index coresponding to the result of a draw which respects the multinomial model defined by l"""
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
        """return a random sequence with a length equal to n corresponding to a HMM"""
        i = HMM.draw_multinomial(self.initial[0])
        m = []
        for j in range(n):
            m.append(HMM.draw_multinomial(self.emissions[i]))
            i = HMM.draw_multinomial(self.transitions[i])
        return m

    def pfw(self, w):
        """return the probability of the sequence w with a particular HMM using fw"""
        n = len(w)
        f = np.zeros((self.nbs,))
        for k in range(self.nbs):
            f[k] = self.initial[0][k] * self.emissions[k][w[0]]
        for i in range(1, n):
            f = np.dot(f, self.transitions) * self.emissions[:,w[i]]
        return f.sum()

    def pbw(self, w):
        """return the probability of the sequence w with a particular HMM using bw"""
        n = len(w)
        b = np.array([1]*self.nbs)
        for i in range(n - 1, 0, -1):
            b = np.dot(self.transitions, self.emissions[:, w[i]] * b)
        p = b * self.initial * self.emissions[:, w[0]]
        return p.sum()

    @staticmethod
    def gen_HMM(nbs, nbl):
        M = HMM
        M.nbl = nbl
        M.nbs = nbs
        init = np.zeros((1, nbs))
        transition = np.zeros((nbs, nbs))
        emission = np.zeros((nbs, nbl))
        for i in range(nbs):
            init[0][i] = random.random()
        init = init / init.sum()
        M.initial = init

        for i in range(nbs):
            for j in range(nbs):
                transition[i][j] = random.random()
        sum_trans = transition.sum(axis=1)
        for i in range(nbs):
            transition[i] = transition[i] / sum_trans[i]
        M.transitions = transition

        for i in range(nbs):
            for j in range(nbl):
                emission[i][j] = random.random()
        sum_emis = emission.sum(axis=1)
        for i in range(nbs):
            emission[i] = emission[i] / sum_emis[i]
        M.emissions = emission

        return M

    def predit(self, w):
        H = self.initial[0]
        for i in range(len(w)):
            H = (self.transitions.T * self.emissions[:, w[i]].T) @ H
        P = []
        for l in range(self.nbl):
            P += [self.emissions[:, l] @ H]
        return P.index(max(P))


###################################

    def viterbi(self, w):
        """
        :param w: Une liste d'observables
        :return: La liste d'états la plus probable correspondant à ce chemin
        """
        chemin_1 = []
        chemin_2 = []
        liste_etats = []
        p_1 = self.initial[0] * self.emissions[:, w[0]]
        p_2 = self.initial[0] * self.emissions[:, w[0]]
        for i in range(self.nbs):
            chemin_1 += [[i]]
            chemin_2 += [[i]]
            liste_etats += [i]
        for i in range(1, len(w)):
            for k in range(self.nbs):
                m = 0
                j_retenu = 0
                for j in range(self.nbs):
                    a = m
                    b = p_1[j] * self.transitions[j, k]
                    m = max(a, b)
                    if m == b:
                        j_retenu = j
                chemin_2[k] = chemin_1[j_retenu] + [k]
                p_2[k] = m * self.emissions[k, w[i]]
            chemin_1 = copy.deepcopy(chemin_2)
            p_1 = copy.deepcopy(p_2)
        return chemin_2[np.argmax(p_2)], np.log(np.max(p_2))

    def logV(self, S):
        ''' calcul de la log vraisemblance'''
        somme = 0
        for w in S:
            somme += np.log(self.pfw(w))
        return somme

    def f(self, w):
        f = np.zeros((self.nbs, len(w)))
        f[:, 0] = self.initial * self.emissions[:, w[0]]
        for i in range(1, len(w)):
            f[:, i] = np.dot(f[:, i - 1], self.transitions) * self.emissions[:, w[i]]
        return f

    def b(self, w):
        b = np.zeros((self.nbs, len(w)))
        b[:, len(w) - 1] = np.array([1] * self.nbs)
        for i in range(len(w) - 2, -1, -1):
            b[:, i] = np.dot(self.transitions, self.emissions[:, w[i + 1]] * b[:, i + 1])
        return b

    def gamma(self, w):
        f = self.f(w)
        b = self.b(w)
        return (f * b) / np.einsum('kt,kt->t', b, f)

    def xi(self, w):
        f = self.f(w)[:, :-1]
        b = self.b(w)[:, 1:]
        emissions = self.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, self.transitions, emissions, b)
        v = np.einsum('kt,kl,lt,lt->t', f, self.transitions, emissions, b)
        somme = np.tile(v, (self.nbs, self.nbs, 1))
        xi = xi / somme
        return xi

    def xi2(self, w):
        f = self.f(w)[:, :-1]
        b = self.b(w)[:, 1:]
        emissions = self.emissions[:, w[1:]]
        xi = np.einsum('kt,kl,lt,lt->klt', f, self.transitions, emissions, b)
        for t in range(xi.shape[2]):
            xi[:, :, t] = xi[:, :, t] / np.sum(xi[:, :, t])
        return xi

    def bw12(self, S):
        if type(S) != list:
            raise TypeError("S doit être une liste")
        if len(S) == 0:
            raise ValueError("S ne doit pas être vide")


        pi = np.zeros(self.nbs)
        for j in range(len(S)):
            pi += np.array(self.gamma(S[j])[:, 0])

        T = np.zeros((self.nbs, self.nbs))
        for j in range(len(S)):
            for t in range(len(S[j]) - 1):
                T += self.xi(S[j])[:, :, t]

        O = np.zeros((self.nbs, self.nbl))
        for j in range(len(S)):
            gamma = self.gamma(S[j])
            for t in range(len(S[j])):
                O[:, S[j][t]] += gamma[:, t]

        maj = HMM(self.nbl, self.nbs, np.array([pi / pi.sum()]), (T.T / T.sum(1)).T, (O.T / O.sum(1)).T)
        return maj

    @staticmethod
    def bw1(m0, S):

        if type(S) != list:
            raise TypeError("S doit être une liste")
        if len(S) == 0:
            raise ValueError("S ne doit pas être vide")

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
                :param nbS: Nombre d'états
                :param nbL: Nombre de sommets
                :param S: Liste de Liste d'observables
                :param N: Entier
                :return: Un HMM généré aléatoirement à nbS états et nbL sommets mis à jour N fois grâce à bw1 pour augmenter
                la vraisemblance
                """
        M = HMM.gen_HMM3(nbs, nbl)
        for i in range(N):
            M = HMM.bw1(M, S)
            #print(M)
        return M

    @staticmethod
    def bw3(nbs, nbl, w, n, m):
        """
                :param nbS: Nombre d'états
                :param nbL: Nombre de sommets
                :param S: Liste de Liste d'observables
                :param N: Entier
                :param M: Entier
                :return: Le HHMi avec 0 <= i <= M-1 qui maximise la vraisemblance de S
                """
        Mi = []
        for i in range(m):
            Mi += [HMM.bw2(nbs, nbl, [w], n)]
        return max(Mi, key=lambda x: x.pfw(w))


###################################
    @staticmethod
    def bw3bis(nbS, nbL, w, N, M):
        """
        :param nbS: Nombre d'états
        :param nbL: Nombre de sommets
        :param S: Liste de Liste d'observables
        :param N: Entier
        :param M: Entier
        :return: Le HHMi avec 0 <= i <= M-1 qui maximise la vraisemblance de S
        """
        max_logV = -math.inf
        hmm = None
        for i in range(M):
            h = HMM.bw2(nbS, nbL, [w], N)
            logV = h.logV([w])
            if max_logV < logV:
                max_logV = logV
                hmm = h
        return hmm

    @staticmethod
    def gen_HMM3(nbs, nbl):
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


"""a = HMM(2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.5, 0.5],[0.7, 0.3]]))
print(a.gen_rand(10))
a.save('/home/vincent/Documents/Test_save')"""

b = HMM.load('HMM1.txt')

print(b.pfw([1, 1]))
print(b.pbw([1, 1]))

print(b)

print(b.predit([1,1,1,1,1]))
print(b.predit([0,0,0]))
a = HMM(3, 3, np.array([[1., 0., 0.]]), np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]), np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
print(a.predit([0, 1, 2, 0]))





w = [0,0]
p = b.initial * b.emissions[:,w[0]]
print(p)
print(len(p))
print(b.nbs)

print(w)
v = b.viterbi(w)
print(v)
print(a.viterbi([0]))
print()


print(HMM.bw1(b, [[0, 1], [1,0], [1, 1, 0, 0]]))
k = HMM.gen_HMM3(2, 2)
print("k : ", k)
print(b)
print(HMM.bw1(k, [[0, 1], [1,0], [1, 1, 0, 0]]))


g = HMM.bw2(2, 2, [[0,1], [1,0], [1, 1, 0, 0]], 100)
print("g : ",g)



i = HMM.bw3bis(2, 2, [0,1], 5, 100)
print("i : ", i)

