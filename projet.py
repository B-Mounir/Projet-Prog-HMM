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
        w = []
        for char in mot:
            w += [ord(char) - 97]
        S += [w[:-1]]
    return S


def liste_sequences(S):
    L = []
    for mot in S:
        w = []
        for c in mot:
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
    s = liste_sequences([w])[0]
    HMMs = [HMM_allemand, HMM_espagnol, HMM_anglais]
    Langue = ['Allemand', 'Espagnol', 'Anglais']
    logps = []
    for M in HMMs:
        logp = M.logV([s])
        logps += [logp]
    return Langue[logps.index(max(logps))]


def gen_mots_langue(M, nbIter):
    L = []
    for i in range(nbIter):
        w = M.gen_rand(3 + i % 5)
        L += [w]
    return liste_mots(L)




print("Bienvenue dans notre programme de présentation du projet de programmation sur les Modèles de Markov cachés.")
print("Réalisé par le groupe PULSE composé de AMRAM Yassine, BAZELAIRE Guillaume, BENNADJI Mounir et WEBERT Vincent.",
      "\n")
print("Nous allons ici générer des HMM adaptés à une langue ainsi que générer des mots qui pourrait appartenir à cette"
      "dernière.", "\n")
print("Pour cela, nous devons récupérer dans un fichier texte une liste de mots qui sera notre liste de séquences "
      "d'états observables.")
print("Nous allons donc utiliser le fichier 'anglais2000' qui contient un peu moins de 2000 mots anglais")
anglais2000 = liste_sequences_fichier('anglais2000')
print("Commençons par générer un HMM aléatoir et appliquons lui l'algorithme de Baum-Welch jusqu'à ce que la "
      "vraissemblance de la liste de mots se stabilise.")
print("Nous prendrons un HMM de 20 états.", "\n")
print("Exécution en cours...", "\n", "Cela peut prendre quelques minutes.", "\n\n")


HMM_anglais_v1, nbiter = HMM.HMM.bw2_limite(20, 26, anglais2000, 10, 10)

print("Nous stockons le HMM mis à jour", nbiter, "fois dans le fichier 'HMM_anglais_v1'.")
print("la Log vraissemblance de notre liste de mots anglais est :", HMM_anglais_v1.logV(anglais2000), "\n\n")

print("Comme vous avez pu le remarquer, l'execution de l'algorithme de Baum-Welch demande un temps de clacul important")
print("Nous allons donc utiliser par la suite des HMMs déjà enregistrés que l'on a entrainé pour chaque langue", "\n")

print("Nous savons donc approcher un modèle optimal pour un nombre d'états donné mais nous voulons déterminer le nombre"
      " d'états optimal")
print("Nous utilisons donc l'algorithme de validation croisée pour l'anglais, l'allemand et l'espagnol à l'aide du "
      "script suivant:", "\n")

script = """
anglais2000 = liste_sequences_fichier("anglais2000")
xval_anglais = xval_limite(5, anglais2000, 26, 25, 50, 10, 1, 10)
lvOpt_anglais = xval_anglais[0]
nbSOpt_anglais = xval_anglais[1]
print("lvOpt_anglais :", lvOpt_anglais)
print("nbSOpt_anglais :", nbSOpt_anglais)

allemand2000 = liste_sequences_fichier("allemand2000")
xval_allemand = xval_limite(5, allemand2000, 26, 25, 50, 10, 1, 10)
lvOpt_allemand = xval_allemand[0]
nbSOpt_allemand = xval_allemand[1]
print("lvOpt_allemand :", lvOpt_allemand)
print("nbSOpt_allemand :", nbSOpt_allemand)

espagnol2000 = liste_sequences_fichier("espagnol2000")
xval_espagnol = xval_limite(5, espagnol2000, 26, 25, 50, 10, 1, 10)
lvOpt_espagnol = xval_espagnol[0]
nbSOpt_espagnol = xval_espagnol[1]
print("lvOpt_espagnol :", lvOpt_espagnol)
print("nbSOpt_espagnol :", nbSOpt_espagnol)
"""

print(script, "\n\n")

print("Malheureusement cet algorithme a demandé trop de temps de calcul et nous avons pu le réaliser pour l'anglais "
      "uniquement.")
print("Le nombre d'états optimal renvoyé est de 45, nous prendrons le même pour les 2 autres langues.", "\n")
print("Nous avons donc entrainé un HMM à 45 états pour chaque langue avec plusieurs initialisations pour limiter "
      "l'influence du HMM généré aléatoirement au départ.")
print("Nous avons utilisé le script suivant pour les sauvegarder :", "\n")

script = """
allemand2000 = liste_sequences_fichier("allemand2000")
espagnol2000 = liste_sequences_fichier("espagnol2000")
anglais2000 = liste_sequences_fichier("anglais2000")

HMM_allemand = HMM.HMM.bw4_limite(45, 26, allemand2000, 10, 1, 10)
HMM_allemand.save("HMM_allemand")

HMM_anglais0 = HMM.HMM.bw4_limite(45, 26, anglais2000, 10, 1, 10)
HMM_anglais0.save("HMM_anglais0")

HMM_espagnol0 = HMM.HMM.bw4_limite(45, 26, espagnol2000, 10, 1, 10)
HMM_espagnol0.save("HMM_espagnol0")
"""

print(script, "\n\n")
print("Nous allons donc charger les HMM ainsi générés pour les utiliser.", "\n\n")

HMM_anglais = HMM.HMM.load('HMM_anglais')
HMM_allemand = HMM.HMM.load('HMM_allemand')
HMM_espagnol = HMM.HMM.load('HMM_espagnol')

print("Tout d'abord nous pouvons déterminer la langue la plus probable d'un mot donné:")
print("Nous allons vous donner quelques exemples :", "\n")

print("'deal' est un mot anglais et la langue probable déterminée est :", langue_probable("deal"), "\n")

print("'geht' est un mot allemand et la langue probable déterminée est :", langue_probable("geht"), "\n")

print("'cocinar' est un mot espagnol et la langue probable déterminée est :", langue_probable("cocinar"), "\n")

print("'drama' est un mot anglais et la langue probable déterminée est :", langue_probable("drama"))
print("En effet ce dernier mot est aussi un mot espagnol ce qui explique pourquoi il donne pas la langue attendue")
print("De  même, pour d'autres mots d'origine latine le programme aura tendance à se tromper")

print("\n\n", "Nous pouvons aussi générer des mots qui pourraient appartenir à une langue particulière :")
print("Voici ci dessosu 10 mots générés par langue :", "\n")

print("\nPour l'anglais:\n")
print(gen_mots_langue(HMM_anglais, 10))

print("\nPour l'allemand:\n")
print(gen_mots_langue(HMM_allemand, 10))

print("\nPour l'espagnol:\n")
print(gen_mots_langue(HMM_espagnol, 10))

print("\n\n", "Merci d'avoir utilisé notre programme.")
print("\n", "L'équipe PULSE.")
