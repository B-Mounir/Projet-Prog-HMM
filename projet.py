#################################################################################
# Title : projet.py                                                             #
# Autors : AMRAM Yassine, BAZELAIRE Guillaume, BENNADJI Mounir, WEBERT Vincent  #
# Date : 18-05-07                                                               #
#################################################################################

import numpy as np
import math
import HMM_class as HMM


def xval(nbFolds, S, nbL, nbSMin, nbSMax, nbIter, nbInit):
    n = len(S)
    l = np.random.permutation(n)
    lvOpt = -math.inf
    nbSOpt = 0
    for nbS in range(nbSMin, nbSMax):
        lv = 0
        h = None
        test = None
        for i in range(1, nbFolds+1):
            f1 = int((i-1)*n/nbFolds)
            f2 = int(i*n/nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2, n)]
            test = [S[l[j]] for j in range(f1, f2)]
            h = HMM.HMM.bw3(nbS, nbL, learn, nbIter, nbInit)
        lv += h.logV(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return lvOpt, nbSOpt

def liste_de_sequences(adr):
    file = open(adr)
    S = []
    for mot in file:
        #print(mot)
        w = []
        for char in mot:
            w += [ord(char) - 97]
        S += [w[:-1]]
    return S

def liste_de_sequences2(adr):
    with open(adr, 'r') as file:
        lines = file.readlines()
        S = []
        for i in range (len(lines)):
            j = 0
            w = []
            while lines[i][j] != chr(10) or i == len(lines)-1:
                w += [ord(lines[i][j]) - 97]
            S += [w]
    return S

anglais2000 = liste_de_sequences("anglais2000")
xval = xval(100, anglais2000, 26, 1, 10, 10, 5)
lvOpt = xval[0]
nbSOpt = xval[1]


HMMopt = None
v_max = - math.inf
h = HMM.HMM.bw4(nbSOpt, 26, anglais2000, 10, 3)
v = h.logV(anglais2000)

print("M: ", h, "logV: ", h.logV(anglais2000))

allemand2000 = liste_de_sequences("allemand2000")