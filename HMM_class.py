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

    def pfw(self, w):
        """return the probability of the sequence w with a particular HMM"""
        if isinstance(w,np.ndarray):
            w = w[0]
        n = len(w)
        f = np.zeros((1, self.nbs))
        for k in range(self.nbs):
            f[0][k] = self.initial[0][k] * self.emissions[k][w[0]]
        for i in range(1, n):
            f = np.dot(f, self.transitions) * self.emissions[:,w[i]]
        return f.sum()

    def viterbi(self, w):
        """return the Viterbi path of the sequence w and his probability"""
        n = len(w)
        p = np.zeros(1, self.nbs)
        c = []
        for k in range(self.nbs):
            p[0][k] = (self.initial[0][k] * self.emissions[k][w[0]])
            c.append([k])
        for i in range(1, n):
  
    def predit(self, w):
        """predict the symbol with the highest probability after the sequence w"""
        H = self.initial
        for i in range(len(w)):
            H = np.dot(self.emissions * self.transitions[:, w[i]],H)
        P = []
        for j in range(self.nbl):
            P += np.dot(self.emissions[:, j],H)
        return P.index(max(P))


'''a = HMM(2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.5, 0.5],[0.7, 0.3]]))
print(a.gen_rand(10))
a.save('/home/vincent/Documents/Test_save')'''

b = HMM.load('/home/vincent/Documents/Cours/Semestre 4/Programmation S4/Projet-Prog-HMM/HMM.txt')

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

n=10
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
print(F3)
