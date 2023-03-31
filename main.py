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

# Fonction qui reconstruit le signal
# Etape 2. Reconstitution du signal
def reconstructionSignal(morceau32ms, m, N, valeurs_signal):
    signal_modif = np.zeros(len(valeurs_signal))
    somme_hamming = np.zeros(len(valeurs_signal))
    for i in range(len(morceau32ms)):
        debut_fenetre = i * m
        fin_fenetre = debut_fenetre + N
        signal_modif[debut_fenetre:fin_fenetre] += morceau32ms[i]
        somme_hamming[debut_fenetre:fin_fenetre] += fenetrageHamming(N)

    # On remplace les 0 par 1 pour éviter les divisions par 0
    for i in range(len(somme_hamming)):
        if somme_hamming[i] == 0:
            somme_hamming[i] = 1

    signal_modif = signal_modif / somme_hamming
    return signal_modif, somme_hamming

# Fonction qui calcule le spectre d'amplitude sur une fenêtre de 32ms
# Etape 4. Spectre d'amplitude
# spectre_amplitude[k] = 20.log(|X_k(o)|)
def spectreAmplitude(spectre, fftsize):
    spectre_amplitude_log = 20 * np.log10(np.abs(spectre))
    spectre_amplitude = np.abs(spectre)
    return spectre_amplitude_log, spectre_amplitude

# Fonction qui calcule la transformée de Fourier inverse
# Etape 3. Calcul de la transformée de Fourier inverse
def fourierInverse(fourier):
    signal = []
    for i in range(len(fourier)):
        signal.append(np.real(FFT.ifft(fourier[i], 1024)))
    return signal

# Fonction qui calcule la transformée de Fourier
# Etape 3. Calcul de la transformée de Fourier
def transformerFourier(morceaux):
    fourier = []
    for i in range(len(morceaux)):
        fourier.append(FFT.fft(morceaux[i], 1024))
    return fourier


def main():
    ## Etape 1. Ouverture du fichier wav
    # Récupération de la fréquence d'échantillonnage, du signal et du nombre d'échantillons
    frequence_enchantillonage, valeurs_signal, nb_echantillon = ouvertureWav()
    print("Fréquence d'échantillonnage : ", frequence_enchantillonage)
    print("Nombre d'échantillons : ", nb_echantillon)
    print("Signal : ", valeurs_signal)

    ## Etape 2. Fenetrage de Hamming
    # Variables de découpage (tout les 8ms et fenêtre de 32ms)
    m = 8 * frequence_enchantillonage // 1000
    N = 32 * frequence_enchantillonage // 1000
    morceau32ms = getMorceau32ms(valeurs_signal, m, N)
    print("Morceaux de 32ms : ", morceau32ms)
    # Fenêtre de Hamming
    morceau32ms = fenetrageHammingSignal(morceau32ms, N)

    ## Etape 3. Calcul de la transformée de Fourier
    fourier = transformerFourier(morceau32ms)

    ## Etape 3. Calcul de la transformée de Fourier inverse
    signal = fourierInverse(fourier)

    ## Etape 4. Calcul du spectre d'amplitude
    # Calcul du spectre d'amplitude
    spectre_amplitude_log, spectre_amplitude = spectreAmplitude(fourier, 1024)
    # Transpose le tableau pour avoir les bonnes dimensions
    spectre_amplitude_log = spectre_amplitude_log.T

    ## Etape 5. Pause sur le debruitage
    plt.imshow(spectre_amplitude_log, aspect='auto')
    plt.show()


    ## Etape 6. Spectre de phase


    ## Reconstitution du signal
    signal_modif, somme_hamming = reconstructionSignal(signal, m, N, valeurs_signal)
    print ("Signal modifié : ", signal_modif)
    # Création du fichier wav
    write("resultat.wav", frequence_enchantillonage, np.int16(signal_modif))

if __name__ == "__main__":
    main()



