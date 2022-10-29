# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:44:06 2022

@author: josua
"""

#Importation des modules nécessaires
import numpy as np
import matplotlib.pyplot as plt

#Définition des constantes
Deff = 1E-10
S = 1E-8
k = 4E-9
Ce = 10
R1 = 0.5

#solveur de système matriciel avec les fonctions d'inversions numpy
def inverseur(M1,V1):
    Minv = np.linalg.inv(M1)
    solution = np.dot(Minv,V1)
    return solution


#Fonction de calcul de la solution numérique de l'équation avec le schéma de la question F
#Variables : R rayon, Ntot nombre de noeud, t pas de temps, tour nombre d'itérations en pas de temps
def solution_numérique(R,Ntot):
    
    deltar = R/(Ntot-1) #taille de l'intervalle
    M = np.zeros((Ntot-1, Ntot-1)) #matrice des équations (à inverser)
    
    #Construction de la matrice des équations (termes faisant intervenir les inconnues)
    for i in range(Ntot-3):
        M[i][i] = (i+0.5)*Deff*deltar
        M[i][i+1] = - 2*(i+1)*Deff*deltar - k*(i+1)*deltar**3
        M[i][i+2] = (i + 1.5)*Deff*deltar
    

    M[-2][-1] = - 2*(Ntot-2)*Deff*deltar - k*(Ntot-2)*deltar**3
    M[-2][-2] =  (Ntot-2.5)*Deff*deltar
    
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
        V[i] = 0
    V[-2] = - (Ntot-1.5)*Deff*deltar*Ce
    V[-1] = 0
    
    
    solution = inverseur(M,V)
    
    return solution




#Etude d'une solution approchée de la solution numérique calculée avec un nombre de noeuds Ntot = 100
Rayons1 = [i*R1/999 for i in range(1000)]
Rayons2 = [i*R1/49 for i in range(50)] #On trace seulement 50 points de la solution approximant
numerique = solution_numérique(R1, 1000) #solution numérique
numerique = np.append(numerique,Ce) #ajout de la concentration Ce au noeud 5
approx = np.polyfit(Rayons1, numerique, 4) #Calcul de la solution approximative en un polynôme de degré 4
print(approx)

#solution approchée (le terme de degré 1 est tronqué)
def solution_approx(x):
    return approx[0]*x**4 + approx[1]*x**3 + approx[2]*x**2 + approx[4]

erreur_approx = [abs(numerique[i] - solution_approx(i*R1/999)) for i in range(1000)] #calcul de l'erreur de l'approximation
print(max(erreur_approx))

#Solution approximée pour le tracé
tab_approx = [solution_approx(i*R1/49) for i in range(50)]
plt.plot(Rayons1, numerique, label = 'solution calculée par le schéma numérique')
plt.plot(Rayons2, tab_approx, 'r+', label = 'approximation de la solution')
plt.legend()
plt.title("Profil stationnaire en concentration")
plt.xlabel("rayon (en m)")
plt.ylabel("Concentration (en mol/m^3")
plt.show()


#Implémentation du terme source
def terme_source(r):
    return  (4*approx[2]*Deff - k*approx[4]) + (9*approx[1]*Deff)*r + (16*approx[0]*Deff - k*approx[2])*r**2 - k*approx[1]*r**3 - k*approx[0]*r**4


#Utilisation du schéma numérique pour la résolution du problème proche
def solution_numérique_MNP(R,Ntot):
    
    deltar = R/(Ntot-1) #taille de l'intervalle
    M = np.zeros((Ntot-1, Ntot-1)) #matrice des équations (à inverser)
    
    #Construction de la matrice des équations (termes faisant intervenir les inconnues)
    for i in range(Ntot-3):
        M[i][i] = (i+0.5)*Deff*deltar
        M[i][i+1] = - 2*(i+1)*Deff*deltar - k*(i+1)*deltar**3 
        M[i][i+2] = (i + 1.5)*Deff*deltar
    

    M[-2][-1] = - 2*(Ntot-2)*Deff*deltar - k*(Ntot-2)*deltar**3
    M[-2][-2] =  (Ntot-2.5)*Deff*deltar
    
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
        S = terme_source((i+1)*deltar)
        V[i] = (i+1)*S*deltar**3
    S = terme_source((Ntot-2)*deltar)
    V[-2] =  (Ntot-2)*S*deltar**3 - (Ntot-1.5)*Deff*deltar*solution_approx(R1)
    V[-1] = 9*approx[1]*Deff
    
    
    solution = inverseur(M,V)
    
    
    return solution



#Calcul des erreurs
valeurs = [] #tableau de la taille des éléments du maillage
L1 = [] #tableau de l'erreur L1 pour une taille
L2 = [] #tableau de l'erreur L2 pour une taille
Linf=[] #tableau de l'erreur Linf pour une taille
for i in range(6,1000,100):
    Ntotex = i
    deltarex = R1/(Ntotex-1)
    valeurs.append(deltarex) #ajout de la taille des éléments
    somme1 = 0
    somme2 = 0
    ref = 0
    #Calcul des erreurs pour un Ntot donné
    for j in range(Ntotex-1):
        diff = abs(solution_numérique_MNP(R1, Ntotex)[j] - solution_approx(j*deltarex))
        somme1+=diff
        somme2+=diff**2
        if diff>ref:
            ref=diff
    L1.append((1/(Ntotex-1))*somme1)
    L2.append(np.sqrt((1/(Ntotex-1))*somme2))
    Linf.append(ref)


#Calcul de la pente moyenne
def pentemoyennelog(tab,intervalle):
    N = len(tab)
    somme=0
    for i in range(N-1):
        somme+=(np.log10(tab[i+1])-np.log10(tab[i]))/np.log10(intervalle)
    return (1/N-1)*somme


#Tracé des graphiques
plt.plot(valeurs, L1, "-gs" ,label = 'L1, pente moyenne : ' + str(round(pentemoyennelog(L1,100),3)))
plt.plot(valeurs, L2, "-bs", label = 'L2, pente moyenne : ' + str(round(pentemoyennelog(L2,100),3)))
plt.plot(valeurs, Linf,"-rs", label = 'Linf, pente moyenne : ' + str(round(pentemoyennelog(Linf,100),3)))
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.title("Erreurs L1, L2 et Linf en fonction de la taille du maillage (échelle log-log)")
plt.xlabel("Taille du maillage (m)")
plt.ylabel("Erreur")
plt.grid(True,which="both", linestyle='--')
plt.show()



#Tracé des solutions dans le cas Ntot=100 noeuds
Rayons = [i*R1/99 for i in range(100)]
numerique = solution_numérique_MNP(R1, 100) #solution numérique
numerique = np.append(numerique,solution_approx(R1)) #ajout de la concentration Ce au noeud 5
analytique = [solution_approx(i*R1/99) for i in range(100)] #solution analytique
plt.plot(Rayons, numerique, label = 'solution calculée')
plt.plot(Rayons, analytique, label = 'solution analytique')
plt.legend()
plt.title("Profil stationnaire en concentration")
plt.xlabel("rayon (en m)")
plt.ylabel("Concentration (en mol/m^3")
plt.grid(True,which="both", linestyle='--')
plt.show()
