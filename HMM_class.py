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
        """Modify the vector defining the initial weights"""
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
        """Modify the array defining the transitions weights"""
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
        """Modify the array defining the emissions weights"""
        if not isinstance(x, np.ndarray):
            raise ValueError("Value Error : should be an array")
        elif x.shape != (self.nbs, self.nbl):
            raise ValueError("Value Error : shape should be (nbs, nbl)")
        elif x.dtype != float:
            raise ValueError("Value Error : elements of array should be float")
        for y in x.sum(axis=1):
            if not np.isclose(y, [1.0], 0.001):
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
        """return a random sequence with a lenght equal to n corresponding to a HMM"""
        i = HMM.draw_multinomial(self.initial[0])
        m = []
        for j in range(n):
            m.append(HMM.draw_multinomial(self.emissions[i]))
            i = HMM.draw_multinomial(self.transitions[i])
        return m

    def fw(self, w):
        """"""
        if isinstance(w,np.ndarray):
            w = w[0]
        n = len(w)
        f = np.zeros((1, self.nbs))
        for k in range(self.nbs):
            f[0][k] = self.initial[0][k] * self.emissions[k][w[0]]
        for i in range(1, n):
            f = np.dot(f, self.transitions) * self.emissions[:,w[i]]
        return f

    def pfw(self, w):
        """return the probability of the sequence w with a particular HMM using fw"""
        f = self.fw(w)
        return f.sum()

    def bw(self, w):
        """"""
        n = len(w)
        b = np.zeros((1, self.nbs))
        for k in range(self.nbs):
            b[0][k] = 1
        for i in range(n-1, 0, -1):
            b = np.dot(b, self.transitions) * self.emissions[:, w[i]]
        return b

    def pbw(self, w):
        """return the probability of the sequence w with a particular HMM using bw"""
        b = self.bw(w)
        p = b * self.initial * self.emissions[:, w[0]]
        return p.sum()

    def viterbi(self, w):
        """return the Viterbi path of the sequence w and his probability"""
        n = len(w)
        p = np.zeros((1, self.nbs))
        c = []
        for k in range(self.nbs):
            p[0][k] = (self.initial[0][k] * self.emissions[k][w[0]])
            c.append([k])
            for i in range(1, n):
                M1 = 0
                M2 = 0
                l_max = -1
                max1 = 0
                max2 = 0
                for l in range (self.nbs):
                    M1 = p[0][l] * self.transitions[l][k] * self.emissions[k][w[i]]
                    M2 = p[0][l] * self.transitions[l][k]
                    if M1 > max1:
                        max1 = M1
                    if M2 > max2 :
                        max2 = M2
                        l_max = l
                p[0][k] = max1
                c[k] = (c[l_max] + [k])
        proba = 0
        ind = -1
        for j in range (len(p[0])):
            if p[0][j] > proba:
                proba = p[0][j]
                ind = j

        return (c[ind], proba)

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

    """def logV(self, S):
        L = self.pfw(S)
        L2 =

    def BW1(self, S):
        l = len(S)
        E = np.zeros((self.nbs, self.nbs))
        G = np.zeros((1, self.nbs))
        for t in range(l):
            for i in range(self.nbs):
                for j in range(self.nbs):
                    E[i][j] =
                G[0][i] =

        # réestimer le modele
        self.initial =
        self.transitions =
        self.emissions ="""








"""a = HMM(2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.5, 0.5],[0.7, 0.3]]))
print(a.gen_rand(10))
a.save('/home/vincent/Documents/Test_save')"""

b = HMM.load('HMM.txt')

#print(b)
#print(b.nbl)
#print()
#print(b.nbs)
#print()
#print(b.initial)
#print()
#print(b.transitions)
#print()
#print(b.emissions)
#print()

'''n=10
i = HMM.draw_multinomial(b.initial[0])
print(i)
m = []
print(m)
for j in range(n):
    print("j:", j)
    m.append(HMM.draw_multinomial(b.emissions[i]))
    print(m)
    i = HMM.draw_multinomial(b.transitions[i])
    print(i)
print("fin:", m)

c = b.gen_rand(10)
print("c:", c)
w = np.array([[0, 1]])
print(w)
F1 = b.pfw(w)
print("forwardArray", F1)

z = [1, 1, 1]
print(z)
F2 = b.pfw(z)
print("forwardList", F2)

F3 = b.pfw(c)
print(F3)'''

print(b.viterbi([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))

"""c = np.array([[0.5, 0.5, 0.6, 0.3]])
print(c)
d = c / c.sum()
print(d)
print(d.sum())

e  =  np.array([[0.5, 0.5, 0.1, 0.4],[0.7, 0.3, 0.8, 0.5], [0.9, 0.4, 0.1, 0.4],[0.3, 0.3, 0.2, 0.1]])
sum_trans = e.sum(axis=1)
for i in range(4):
    e[i] = e[i] / sum_trans[i]
    print("somme ligne", i,":",e[i].sum())
print(e)"""

"""M = HMM.gen_HMM(2, 3)
print("nbs", M.nbs)
print("nbl", M.nbl)
print("init", M.initial, M.initial.sum())
print("trans", M.transitions, M.transitions.sum(axis=1))
print("emis", M.emissions, M.emissions.sum(axis=1))"""

print(b.pfw([1, 1]))
print(b.pbw([1, 1]))

print(b)

print(b.predit([1,1,1,1,1]))
print(b.predit([0,0,0]))
a = HMM(3, 3, np.array([[1., 0., 0.]]), np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]), np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
print(a.predit([0, 1, 2, 0]))
