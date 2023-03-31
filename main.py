from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import numpy.fft as FFT

# Fonction qui ouvre un fichier wav et
# renvoie la fréquence d'échantillonnage, le signal et le nombre d'échantillons
# Etape 1. Ouverture du fichier wav
def ouvertureWav(filename = 'fichiers_bruit/test_seg.wav'):
    fichier = filename
    frequence_enchantillonage, valeurs_signal = read(fichier)
    nb_echantillon = valeurs_signal.shape[0]
    duree_ms = 1000 * nb_echantillon / frequence_enchantillonage

    return frequence_enchantillonage, valeurs_signal, nb_echantillon

# Fonction qui renvoie une fenêtre de Hamming
# Etape 2. Fenetrage de Hamming
def fenetrageHamming(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

# Fonction qui renvoie le fenetrage de Hamming d'un signal de taille N
def fenetrageHammingSignal(signal, N):
    for i in range(len(signal)):
        signal[i] = signal[i] * fenetrageHamming(N)
    return signal

# Fonction qui renvoie un tableau de morceaux de 32ms
# Etape 2. Récupération de la fenêtre à l'instant i et de taille m
def getMorceau32ms(signal, m, N):
    nb_fenetres = int((len(signal) - N) / m) + 1
    m32ms = np.zeros((nb_fenetres, N))
    for i in range(nb_fenetres):
        debut_fenetre = i * m
        fin_fenetre = debut_fenetre + N
        m32ms[i] = signal[debut_fenetre:fin_fenetre]
    return m32ms

def main():
    ## Etape 1. Ouverture du fichier wav
    # Récupération de la fréquence d'échantillonnage, du signal et du nombre d'échantillons
    frequence_enchantillonage, valeurs_signal, nb_echantillon = ouvertureWav()
    print("Fréquence d'échantillonnage : ", frequence_enchantillonage)
    print("Nombre d'échantillons : ", nb_echantillon)
    print("Signal : ", valeurs_signal)

    ## Etape 2. Fenetrage de Hamming
    # Variables de découpage (tout les 8ms et fenêtre de 32ms)
    m = 8
    N = 32
    morceau32ms = getMorceau32ms(valeurs_signal, m, N)
    print("Morceaux de 32ms : ", morceau32ms)
    # Fenêtre de Hamming
    morceau32ms = fenetrageHammingSignal(morceau32ms, N)

    ## Reconstitution du signal
    signal_modif = np.zeros(len(valeurs_signal))
    somme_hamming = np.zeros(len(valeurs_signal))
    for i in range(len(morceau32ms)):
        debut_fenetre = i * m
        fin_fenetre = debut_fenetre + N
        signal_modif[debut_fenetre:fin_fenetre] += morceau32ms[i]
        somme_hamming[debut_fenetre:fin_fenetre] += fenetrageHamming(N)

    for i in range(len(somme_hamming)):
        if somme_hamming[i] == 0:
            somme_hamming[i] = 1
    signal_modif = signal_modif / somme_hamming
    print ("Signal modifié : ", signal_modif)
    write("resultat.wav", frequence_enchantillonage, np.int16(signal_modif))

if __name__ == "__main__":
    main()



