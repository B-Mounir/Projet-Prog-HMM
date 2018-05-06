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
            h = HMM.HMM.bw3(nbL, nbS, learn, nbIter, nbInit)
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
            while lines[i][j] != "/n" or i == len(lines)-1:
                w += [ord(lines[i][j]) - 97]
            S += [w]
    return S
print(liste_de_sequences("anglais2000"))
#print(liste_de_sequences2("anglais2000"))


anglais2000 = liste_de_sequences("anglais2000")
"""
HMMopt = None
v_max = - math.inf
for n in range(1, 27):
    h = HMM.HMM.bw4(n, 26, anglais2000, 5, 1)
    v = h.logV(anglais2000)
    if v_max < v:
        v_max = v
        HMMopt = h
print(HMMopt)"""
M = HMM.HMM(26, 4, np.array([[ 0.11159716,  0.09084701,  0.64593771,  0.15161812]]), np.array([[ 0.03725947,  0.10732655,  0.38337422,  0.47203976],
       [ 0.30689375,  0.02854844,  0.4335592 ,  0.23099861],
       [ 0.0105015 ,  0.50054992,  0.04870977,  0.44023881],
       [ 0.58085295,  0.2223527 ,  0.10467856,  0.09211579]]), np.array([[  3.68936047e-02,   5.99469471e-03,   4.98889341e-02,
          4.22547303e-02,   4.35846603e-02,   1.70268482e-02,
          6.33738435e-03,   9.01047718e-03,   8.38736327e-02,
          7.82228774e-04,   8.47872639e-03,   3.73464655e-02,
          3.43529418e-02,   1.93357581e-01,   5.98728122e-03,
          2.94433104e-02,   5.32471183e-03,   2.54205626e-01,
          2.02824194e-03,   4.64195867e-02,   1.47878197e-02,
          2.53768916e-02,   1.30910100e-02,   3.82426243e-03,
          2.88628388e-02,   1.46550976e-03],
       [  1.36308065e-01,   2.84123240e-04,   8.98533523e-03,
          1.26376705e-02,   2.28073141e-01,   2.67236363e-04,
          2.60877099e-02,   3.46003295e-02,   1.35828928e-01,
          1.38142962e-04,   5.07804006e-03,   9.52328889e-02,
          7.49051182e-04,   4.91697565e-02,   7.50425163e-02,
          1.14581884e-02,   1.52524699e-03,   3.25447069e-02,
          7.11602357e-02,   3.66563037e-02,   2.88598112e-02,
          7.46266392e-04,   3.80532720e-03,   3.86765224e-03,
          8.87778546e-04,   5.54790112e-06],
       [  3.98497896e-02,   3.66654615e-02,   9.71440553e-02,
          8.19116501e-02,   4.33406874e-02,   3.99491031e-02,
          5.75813096e-02,   2.74788793e-02,   4.85823203e-03,
          7.62461998e-03,   6.84821255e-03,   3.22229664e-02,
          6.84482140e-02,   5.73313562e-02,   1.55068117e-02,
          3.87169246e-02,   1.46106972e-03,   2.40263031e-02,
          1.62644102e-01,   8.57905372e-02,   1.26049187e-02,
          1.77417178e-02,   2.74639383e-02,   1.37250130e-03,
          1.13091818e-02,   1.07456027e-04],
       [  1.01336139e-01,   1.21224128e-03,   1.07226627e-02,
          2.67892808e-02,   2.12968087e-01,   7.79571672e-03,
          2.79538025e-03,   1.88981107e-02,   9.73924224e-02,
          7.85651114e-05,   9.39206715e-03,   3.01850047e-02,
          5.49785496e-03,   6.90027162e-03,   1.63862676e-01,
          3.38591025e-02,   2.16069640e-04,   8.67806030e-03,
          4.84745008e-02,   1.19161105e-01,   6.23443187e-02,
          7.10392721e-03,   8.60700700e-04,   3.56279610e-03,
          1.98947672e-02,   1.81711542e-05]]))

print(M.predit([1, 4, 2, 14, 12, 8, 13]))

allemand2000 = liste_de_sequences("allemand2000")