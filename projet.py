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
        #h = None
        #test = None
        for i in range(1, nbFolds+1):
            f1 = int((i-1)*n/nbFolds)
            f2 = int(i*n/nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2, n)]
            test = [S[l[j]] for j in range(f1, f2)]
            h = HMM.HMM.bw4(nbS, nbL, learn, nbIter, nbInit)
            lv += h.logV(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return lvOpt, nbSOpt

def xval_limite(nbFolds, S, nbL, nbSMin, nbSMax, limite, tolerance, nbInit):
    n = len(S)
    l = np.random.permutation(n)
    lvOpt = -math.inf
    nbSOpt = 0
    for nbS in range(nbSMin, nbSMax):
        lv = 0
        #h = None
        #test = None
        for i in range(1, nbFolds+1):
            f1 = int((i-1)*n/nbFolds)
            f2 = int(i*n/nbFolds)
            learn = [S[l[j]] for j in range(f1)]
            learn += [S[l[j]] for j in range(f2, n)]
            test = [S[l[j]] for j in range(f1, f2)]
            h = HMM.HMM.bw4_limite(nbS, nbL, learn, limite, tolerance, nbInit)
            lv += h.logV(test)
        if lv > lvOpt:
            lvOpt = lv
            nbSOpt = nbS
    return lvOpt, nbSOpt

def liste_sequences_fichier(adr):
    file = open(adr)
    S = []
    for mot in file:
        #print(mot)
        w = []
        for char in mot:
            w += [ord(char) - 97]
        S += [w[:-1]]
    return S

def liste_sequences(S):
    L = []
    for w in S:
        w = []
        for c in w:
            w += [ord(c) - 97]
        w = w[:-1]
        L += [w]
    return L

def liste_mots(S):
    L = []
    for w in S:
        mot = ''
        for c in w:
            char = chr(c + 97)
            mot += char
        L += [mot]
    return L

def langue_probable(w):
    s = liste_sequences([w])
    HMMs = [HMM_allemand, HMM_espagnol, HMM_anglais]
    Langue = ['Allemand', 'Espagnol', 'Anglais']
    logps = []
    for M in HMMs:
        logp = M.log_vraissemblance([s])
        logps += [logp]
    return Langue[logps.index(max(logps))]

allemand2000 = liste_sequences_fichier("allemand2000")
espagnol2000 = liste_sequences_fichier("espagnol2000")
anglais2000 = liste_sequences_fichier("anglais2000")

HMM_allemand = HMM.HMM.bw4_limite(45, 26, allemand2000, 10, 1, 10)
HMM_espagnol = HMM.HMM.bw4_limite(45, 26, espagnol2000, 10, 1, 10)
HMM_anglais = HMM.HMM.bw4_limite(45, 26, anglais2000, 10, 1, 10)

print("HMM_allemand :", HMM_allemand)
print()
print("HMM_espagnol :", HMM_espagnol)
print()
print("HMM_anglais :", HMM_anglais)
