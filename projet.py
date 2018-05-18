#################################################################################
# Title : projet.py                                                             #
# Autors : AMRAM Yassine, BAZELAIRE Guillaume, BENNADJI Mounir, WEBERT Vincent  #
# Date : 18-05-07                                                               #
#################################################################################

import numpy as np
import math
import HMM_class as HMM


def xval(nbFolds, S, nbL, nbSMin, nbSMax, nbIter, nbInit):
    """
    :param nbFolds: Number of folds (Integer)
    :param S: list of observable states sequences
    :param nbL: Number of letters (Integer)
    :param nbSMin: Number of states min (Integer)
    :param nbSMax: Number of states max (Integer)
    :param nbIter: Number of iterations of bw2 (Integer)
    :param nbInit: Number of initilisations in bw4 (Integer)
    :return: The optimal number of states for an HMM adapted to S and le log likelihood of S with this HMM
    """
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
    """
    :param nbFolds:  Number of folds (Integer)
    :param S: List of observable states sequences
    :param nbL:  Number of letters (Integer)
    :param nbSMin: Number of states min (Integer)
    :param nbSMax: Number of states max (Integer)
    :param limite: Number of iterations where the log likelihood has to be stable (Integer)
    :param tolerance: Tolerance of the log likelihood stabilisation (Float or Integer)
    :param nbInit: Number of initilisations in bw4 (Integer)
    :return: The optimal number of states for an HMM adapted to S and le log likelihood of S with this HMM
    """
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
    """
    :param adr: language text file adress (String)
    :return: The list of observable states sequences represented by the a language text file
    """
    file = open(adr)
    S = []
    for mot in file:
        w = []
        for char in mot:
            w += [ord(char) - 97]
        S += [w[:-1]]
    return S


def liste_sequences(S):
    """
    :param S: List of words (List of strings)
    :return: The list of observable states sequences represented by the list of words S
    """
    L = []
    for mot in S:
        w = []
        for c in mot:
            w += [ord(c) - 97]
        w = w[:-1]
        L += [w]
    return L


def liste_mots(S):
    """
    :param S: List of observable states sequences (List of lists)
    :return: The list of words represented by S
    """
    L = []
    for w in S:
        mot = ''
        for c in w:
            char = chr(c + 97)
            mot += char
        L += [mot]
    return L


def langue_probable(w):
    """
    :param w: word (String)
    :return: the most probable language of the word w
    """
    s = liste_sequences([w])[0]
    HMMs = [HMM_allemand, HMM_espagnol, HMM_anglais]
    Langue = ['Allemand', 'Espagnol', 'Anglais']
    logps = []
    for M in HMMs:
        logp = M.logV([s])
        logps += [logp]
    return Langue[logps.index(max(logps))]


def gen_mots_langue(M, nbIter):
    """
    :param M: HMM of a language (HMM)
    :param nbIter: Number of words generated (Integer)
    :return: a list of nbIter words generated with the language HMM M
    """
    L = []
    for i in range(nbIter):
        w = M.gen_rand(3 + i % 5)
        L += [w]
    return liste_mots(L)




print("Bienvenue dans notre programme de présentation du projet de programmation sur les Modèles de Markov cachés.",
      "\nRéalisé par le groupe PULSE composé de AMRAM Yassine, BAZELAIRE Guillaume, BENNADJI Mounir et WEBERT "
            "Vincent.","\n")
input("Appuyez sur ENTRÉE pour continuer.\n")


var = input("Voulez vous tester les fonctionnalités de la classe HMM ? Si non nous pouvons passer directement à la "
            "demonstration du projet (o/n) :")
while var not in ['o', 'n']:
    var = input("Saisie invalide. Voulez vous tester les fonctionnalités de la classe HMM ? (o/n)")

while var == 'o' :
    print("Nous allons donc tester les fonctionnalités de la classe HMM.")
    while True:
        adr = input("Tout d'abord, entrez l'adresse d'un fichier text contenant un HMM de votre choix :")
        try:
            M = HMM.HMM.load(adr)
            break
        except:
            print("Adresse invalide")
    print("Ce HMM est maintenant chargé et nous pouvons l'utiliser.\n")

    print("Commençons par générer une séquence d'états observables :")
    l = int(input("Veuillez entrer la longueur de la séquence voulue :"))
    w = M.gen_rand(l)
    print("\nLa sequences générée est :", w)
    print("La probabilité de cette séquence avec notre HMM est :", M.pfw(w))
    c, p = M.viterbi(w)
    print("Le chemin qui maximise la probabilité d'obtenir ce mot est", c, "et la log probabilité est alors de", p)
    char = M.predit(w)
    print("L'état observable le plus probable d'être émis si on continuait serait :", char)

    print("\n\nMaintenant générerons une liste de séquences d'états observables :")
    n = int(input("veuillez entrer le nombre de séquences voulu :"))
    l = int(input("Veuillez entrer la longueur de la séquence voulue :"))
    S = []
    for i in range (n):
        w = M.gen_rand(l)
        S += [w]
    print("\nLa liste de sequences générée est :", S)
    print("La log vraisemblance de cette liste de séquence avec notre HMM est :", M.logV(S))
    print("\nNous allons maintenant exécuter Baum-Welch sur notre HMM")
    m = int(input("Veuillez entrer le nombre de fois que vous vous exécuter Baul-Welch :"))
    for j in range(m):
        M = HMM.HMM.bw1(M, S)
    print("\nLe nouveau HMM mis à jour par BW a donc été généré.")
    print("La log vraisemblance de la liste de séquences est maintenant de :", M.logV(S))
    print("Nous devrions constater qu'elle a augmenté")


    var = input("Voulez vous tester les fonctionnalités de la classe HMM à nouveau ? (o/n) :")
    while var not in ['o', 'n']:
        var = input("Saisie invalide. Voulez vous tester les fonctionnalités de la classe HMM à nouveau? (o/n)")



print("\n\n\nNous allons maintenant générer des HMM adaptés à une langue ainsi que générer des mots qui pourrait appartenir à cette dernière.", "\n")
print("Pour cela, nous devons récupérer dans un fichier texte une liste de mots qui sera notre liste de séquences "
      "d'états observables.")
print("Nous allons donc utiliser le fichier 'anglais2000' qui contient un peu moins de 2000 mots anglais")
anglais2000 = liste_sequences_fichier('anglais2000')
print("Commençons par générer un HMM aléatoire et appliquons lui l'algorithme de Baum-Welch jusqu'à ce que la "
      "vraissemblance de la liste de mots se stabilise.")
print("Nous prendrons un HMM de 20 états.", "\n")
print("Exécution en cours...", "\nCela peut prendre quelques minutes.", "\n\n")


HMM_anglais_v1, nbiter = HMM.HMM.bw2_limite(20, 26, anglais2000, 10, 10)

print("Nous stockons le HMM mis à jour", nbiter, "fois dans le fichier 'HMM_anglais_v1'.")
HMM_anglais_v1.save('HMM_anglais_v1')
print("la Log vraisemblance de notre liste de mots anglais est :", HMM_anglais_v1.logV(anglais2000), "\n\n")

print("Comme vous avez pu le remarquer, l'execution de l'algorithme de Baum-Welch demande un temps de calcul important")
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
print("Le nombre d'états optimal renvoyé est de 45, nous prendrons le même pour les 2 autres langues bien que nombre de"
      " grande chance d'être différent, il ne devrait pas être très éloigné.", "\n")
print("Nous avons donc entrainé un HMM à 45 états pour chaque langue avec plusieurs initialisations pour limiter "
      "l'influence du HMM généré aléatoirement au départ.")
print("Nous avons utilisé le script suivant pour les sauvegarder :", "\n")

script = """
allemand2000 = liste_sequences_fichier("allemand2000")
espagnol2000 = liste_sequences_fichier("espagnol2000")
anglais2000 = liste_sequences_fichier("anglais2000")

HMM_allemand = HMM.HMM.bw4_limite(45, 26, allemand2000, 10, 1, 10)
HMM_allemand.save("HMM_allemand")

HMM_anglais = HMM.HMM.bw4_limite(45, 26, anglais2000, 10, 1, 10)
HMM_anglais.save("HMM_anglais")

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
print("En effet ce dernier mot est aussi un mot espagnol ce qui explique pourquoi il donne pas la langue attendue.")
print("De  même, pour d'autres mots d'origine latine le programme aura tendance à se tromper.")

print("\n\n", "Nous pouvons aussi générer des mots qui pourraient appartenir à une langue particulière :")
print("Voici ci dessosu 10 mots générés par langue :", "\n")

print("\nPour l'anglais:\n")
print(gen_mots_langue(HMM_anglais, 10))

print("\nPour l'allemand:\n")
print(gen_mots_langue(HMM_allemand, 10))

print("\nPour l'espagnol:\n")
print(gen_mots_langue(HMM_espagnol, 10))

print("\n\nMerci d'avoir utilisé notre programme.")
print("\nL'équipe PULSE.")
