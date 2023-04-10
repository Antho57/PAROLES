from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import numpy.fft as FFT


# Etape 1. Ouverture du fichier wav
def ouvertureWav(filename = 'fichiers_bruit/test_seg_bruit_10dB.wav'):
    fichier = filename
    frequence_enchantillonage, valeurs_signal = read(fichier)
    nb_echantillon = valeurs_signal.shape[0]
    duree_ms = 1000 * nb_echantillon / frequence_enchantillonage

    return frequence_enchantillonage, valeurs_signal, nb_echantillon

# Etape 2. Fenetrage de Hamming
def fenetrageHamming(N):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

# Fonction qui renvoie le fenetrage de Hamming d'un signal de taille N
def fenetrageHammingSignal(signal, N):
    for i in range(len(signal)):
        signal[i] = signal[i] * fenetrageHamming(N)
    return signal

# Etape 2. Récupération de la fenêtre à l'instant i et de taille m
def getMorceau32ms(signal, m, N):
    nb_fenetres = int((len(signal) - N) / m) + 1
    m32ms = np.zeros((nb_fenetres, N))
    for i in range(nb_fenetres):
        debut_fenetre = i * m
        fin_fenetre = debut_fenetre + N
        m32ms[i] = signal[debut_fenetre:fin_fenetre]
    return m32ms

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

## Etape 3. Calcul de la transformée de Fourier
def transformerFourier(morceaux):
    fourier = []
    for i in range(len(morceaux)):
        fourier.append(FFT.fft(morceaux[i], 1024))
    return fourier

## Etape 3. Fourier inverse
def fourierInverseDebruitee(fourier_inverse_debruitee):
    signal = []
    for i in range(len(fourier_inverse_debruitee)):
        signal.append(np.real(FFT.ifft(fourier_inverse_debruitee[i], 1024))[:32])
    return signal

#Etape 4. Calcul de l'amplitude du spectre et le log
def spectreAmplitude(fourier):
    spectre = []
    for i in range(len(fourier)):
        spectre.append(np.abs(fourier[i]))
    return spectre

## Etape 6. Calcul de la phase
def spectrePhase(fourier):
    spectre = []
    for i in range(len(fourier)):
        spectre.append(np.angle(fourier[i]))
    return spectre

## Etape 7. Reconstruction du signal
def signalDebruite(spectre_debruite, spectre_phase):
    fourier_inverse_debruitee = []
    for i in range(len(spectre_debruite)):
        fourier_inverse_debruitee.append(spectre_debruite[i] * np.exp(1j * spectre_phase[i]))
    return fourier_inverse_debruitee

## Etape 8. Calcul de la moyenne des 5 premiers spectres
def moyenneSpectreAmplitudeBruit(spectre, n_frames=5):
    return np.mean(spectre[:n_frames], axis=0)

## Etape 9. Débruitage par soustraction
def debruitage(spectre, moyenne, alpha=2, beta=1, gamma=0):
    diff = spectre - (alpha * (moyenne - beta))
    np.maximum(diff, gamma * np.abs(spectre), out=diff)
    return diff


#Affichage du signal original et du signal modifié
def affichageSignal(signal, signal_modif):
    plt.figure(1)
    plt.subplot(311)
    plt.plot(signal)
    plt.title("Signal original")
    plt.subplot(313)
    plt.plot(signal_modif)
    plt.title("Signal modifié")
    plt.show()


def main():
    ## Etape 1. Ouverture du fichier wav
    # Récupération de la fréquence d'échantillonnage, du signal et du nombre d'échantillons
    frequence_enchantillonage, valeurs_signal, nb_echantillon = ouvertureWav()

    ## Etape 2. Fenetrage de Hamming
    # Variables de découpage (tout les 8ms et fenêtre de 32ms)
    m = 8
    N = 32
    morceau32ms = getMorceau32ms(valeurs_signal, m, N)
    # Fenêtre de Hamming
    morceau32ms = fenetrageHammingSignal(morceau32ms, N)

    ## Etape 3. Calcul de la transformée de Fourier
    fourier = transformerFourier(morceau32ms)

    ## Etape 4. Calcul du spectre d'amplitude
    spectre = spectreAmplitude(fourier)

    ## Etape 5. Pause sur le spectre d'amplitude
    spectre_log = (20 * np.log10(np.abs(fourier))).T
    plt.imshow(spectre_log, aspect='auto')
    plt.show()

    ## Etape 6. Calcul du spectre de phase
    spectre_phase = spectrePhase(fourier)

    ## Etape 8. Calcul de la moyenne des spectres d'amplitude du bruit
    moyenne = moyenneSpectreAmplitudeBruit(spectre)

    ## Etape 9. Débruitage
    spectre_debruite = debruitage(spectre, moyenne)

    ## Etape 7. Calcul du signal débruité
    fourier_inverse_debruitee = signalDebruite(spectre_debruite, spectre_phase)

    ## Etape 3. Calcul de la transformée de Fourier inverse débruitée
    signal_debruite = fourierInverseDebruitee(fourier_inverse_debruitee)

    ## Affichage du signal original et du signal débruité
    affichageSignal(morceau32ms, signal_debruite)

    ##Reconstruction du signal
    signal_modif, somme_hamming = reconstructionSignal(signal_debruite, m, N, valeurs_signal)

    ##Création du fichier wav
    write("resultat.wav", frequence_enchantillonage, np.int16(signal_modif))

if __name__ == "__main__":
    main()