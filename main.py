from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np

def ouverture_wav():
    fichier = 'test_seg.wav'
    frequence_enchantillonage, valeurs_signal = read(fichier)
    nb_echantillon = valeurs_signal.shape[0]
    duree_ms = 1000 * nb_echantillon / frequence_enchantillonage

    return frequence_enchantillonage, valeurs_signal, nb_echantillon

# fonction qui renvoie une fenêtre de Hamming
def fenetrageHamming(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

# fonction qui renvoie un tableau de morceaux de 32ms
def morceaux_32ms(signal, m, N):
    nb_fenetres = int((len(signal) - N) / m) + 1
    morceaux_32ms = np.zeros((nb_fenetres, N))
    for i in range(nb_fenetres):
        debut_fenetre = i * m
        fin_fenetre = debut_fenetre + N
        morceaux_32ms[i] = signal[debut_fenetre:fin_fenetre]
    return morceaux_32ms


frequence_enchantillonage, valeurs_signal, nb_echantillon = ouverture_wav()
print(frequence_enchantillonage)
print(nb_echantillon)
print(valeurs_signal)

#decoupage toutes les 8ms du signal
m = 8
N = 32
print("<----------------------->")

morceaux_32ms = morceaux_32ms(valeurs_signal, m, N)
print(morceaux_32ms)

#fenetrage de hamming
for i in range(len(morceaux_32ms)):
    morceaux_32ms[i] = morceaux_32ms[i] * fenetrageHamming(N)

#Reconstruction du signal
signal_modif = np.zeros(len(valeurs_signal))
somme_hamming = np.zeros(len(valeurs_signal))
for i in range(len(morceaux_32ms)):
    debut_fenetre = i * m
    fin_fenetre = debut_fenetre + N
    signal_modif[debut_fenetre:fin_fenetre] += morceaux_32ms[i]
    somme_hamming[debut_fenetre:fin_fenetre] += fenetrageHamming(N)

print("<----------------------->")
print(somme_hamming)
print("<----------------------->")
print(signal_modif)
print("<----------------------->")
signal_modif = signal_modif / somme_hamming

print("<----------------------->")
print(signal_modif)




