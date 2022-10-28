# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 01:56:25 2022

@author: josua
"""

#Importation des modules nécessaires
import numpy as np
import matplotlib.pyplot as plt

#Définition des constantes
Deff = 1E-10
S = 1E-8
Ce = 10
R1 = 0.5
C0 = 5
t0 = 1

#solveur de système matriciel avec les fonctions d'inversions numpy
def inverseur(M1,V1):
    Minv = np.linalg.inv(M1)
    solution = np.dot(Minv,V1)
    return solution

#Fonction de calcul de la solution numérique de l'équation avec le schéma de la question F
#Variables : R rayon, Ntot nombre de noeud, t pas de temps, tour nombre d'itérations en pas de temps
def solution_numérique(R,Ntot,t,tour):
    
    deltar = R/(Ntot-1) #taille de l'intervalle
    M = np.zeros((Ntot-1, Ntot-1)) #matrice des équations (à inverser)

    #Construction de la matrice des équations (termes faisant intervenir les inconnues)
    for i in range(Ntot-3):
        M[i][i] = (i+0.5)*Deff*t*deltar
        M[i][i+1] =  -(i+1)*deltar**3 - 2*(i+1)*Deff*t*deltar
        M[i][i+2] = (i+1.5)*Deff*t*deltar
        
    M[-2][-1] = -(Ntot-2)*deltar**3 - 2*(Ntot-2)*Deff*t*deltar #(derniers termes de la matrice)
    M[-2][-2] =  (Ntot-2.5)*Deff*t*deltar
        
#Même condition que précédemment    
    M[-1][0] = -1
    M[-1][1] = 1
    M[-1][2] = 0
    
#Condition avec Schéma de Gear
    #M[-1][0] = 3
    #M[-1][1] = -4
    #M[-1][2] = 1
        
    #Construction du vecteur (terme faisant intervenir les données et variables déjà calculées)
    V = np.zeros(Ntot-1)
    for i in range(Ntot-3):
        S = np.exp(-t/t0)*(C0*Deff*4/R**2 + (C0/t0)*(((i+1)*deltar)**2/R**2 - 1))
        V[i] = (i+1)*S*t*deltar**3 - (C0*(((i+1)*deltar)**2/R**2 - 1)*np.exp(-t/t0)+Ce)*(i+1)*deltar**3
    S = np.exp(-t/t0)*(C0*Deff*4/R**2 + (C0/t0)*(((Ntot-2)*deltar)**2/R**2 - 1))
    V[-2] = (Ntot-2)*S*t*deltar**3 - (Ntot-1.5)*Deff*deltar*t*Ce - (C0*(((Ntot-2)*deltar)**2/R**2 - 1)*np.exp(-t/t0)+Ce)*(Ntot-2)*deltar**3
    V[-1] = 0


    #Résolution du système matriciel par itérations
    for j in range(tour):
        solution = inverseur(M,V)
        V = np.zeros(Ntot-1)
        #Actualisation du vecteur V avec les résultats obtenus pour la prochaine itération (termes en c^t)
        for i in range(Ntot-3):
            S = np.exp(-t*(j+2)/t0)*(C0*Deff*4/R**2 + (C0/t0)*(((i+1)*deltar)**2/R**2 - 1))
            V[i] = (i+1)*S*t*deltar**3 - solution[i+1]*(i+1)*deltar**3
        S = np.exp(-t*(j+2)/t0)*(C0*Deff*4/R**2 + (C0/t0)*(((Ntot-2)*deltar)**2/R**2 - 1))
        V[-2] = (Ntot-2)*S*t*deltar**3 - (Ntot-1.5)*Deff*t*deltar*Ce - solution[Ntot-2]*(Ntot-2)*deltar**3
        V[-1] = 0
    
    
    return solution



#Fonction solution analytique
def solution_analytique(t,r,R):
    return C0*(r**2/R**2 - 1)*np.exp(-t/t0) + Ce

#Test de la fonction solveur numérique
#On fixe les variables tel que voulu
Ntot1 = 100
t = 0.0001 #pas de temps = jour
print(solution_numérique(R1,Ntot1,t,10))
print(solution_analytique(10*t,0,R1))


#Calcul des erreurs
valeurs = [i*(R1/(Ntot1-1)) for i in range(Ntot1)]
valeurs2 = [i*(R1/(Ntot1-1)) for i in range(Ntot1-1)]
valeursnum = [solution_numérique(R1,Ntot1,t,10)[i] for i in range(Ntot1-1)]
valeursex = [solution_analytique(10*t,i*(R1/(Ntot1-1)), R1) for i in range(Ntot1)]

valeursnum.append(Ce)
diff = [abs(valeursnum[i]-valeursex[i]) for i in range(1,Ntot1)]

#Tracé des graphiques
plt.plot(valeurs, valeursnum, label = 'Solution calculée')
plt.plot(valeurs, valeursex, label = 'Solution analytique')
plt.legend()
plt.title("Tracé de la concentration à un temps donné (MMS)")
plt.xlabel("Rayon")
plt.ylabel("Concentration")
plt.grid(linestyle='--')
plt.show()


plt.plot(valeurs2, diff, label = 'Erreur')
plt.legend()
plt.title("Tracé de l'erreur'")
plt.xlabel("Rayon")
plt.ylabel("Erreur")
plt.grid(linestyle='--')
plt.show()


#test stationnaire
Ntot2=50
mem = [0 for i in range(Ntot2)]
tourmax = 1
truediff = 1
while truediff > 0.01 :
    valeursex = [solution_analytique(tourmax*t,i*(R1/(Ntot2-1)), R1) for i in range(Ntot2)]
    difference = [abs(valeursex[i]-mem[i]) for i in range(Ntot2)]
    truediff = max(difference)
    mem = valeursex
    tourmax+=100
print(tourmax)


#Calcul des erreurs
for tour in range(1,tourmax,int(0.05*tourmax)) :
    valeurs = [] #tableau de la taille des éléments du maillage
    L1 = [] #tableau de l'erreur L1 pour une taille
    L2 = [] #tableau de l'erreur L2 pour une taille
    Linf=[] #tableau de l'erreur Linf pour une taille
    for i in range(6,100,10):
        Ntotex = i
        deltarex = R1/(Ntotex-1)
        valeurs.append(deltarex) #ajout de la taille des éléments
        somme1 = 0
        somme2 = 0
        ref = 0
        #Calcul des erreurs pour un Ntot donné
        for j in range(Ntotex-1):
            diff = abs(solution_numérique(R1, Ntotex,t,tour)[j] - solution_analytique(tour*t,j*deltarex,R1))
            somme1+=diff
            somme2+=diff**2
            if diff>ref:
                ref=diff
        L1.append((1/(Ntotex-1))*somme1)
        L2.append(np.sqrt((1/(Ntotex-1))*somme2))
        Linf.append(ref)
        
    #Tracé des graphiques
    plt.plot(valeurs, L1, label = 'L1')
    plt.plot(valeurs, L2, label = 'L2')
    plt.plot(valeurs, Linf, label = 'Linf')
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Erreurs L1, L2 et Linf en fonction de la taille du maillage (échelle log-log), nombre de tours" + str(tour))
    plt.xlabel("Taille du maillage (10^xx m)")
    plt.ylabel("Erreur (10^xx)")
    plt.grid(True,which="both", linestyle='--')
    plt.show()
