import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def plan_de_test_A(amplitude = 1, frequency_test  = 1000, fe = 20000, N = 768, plot_sinus = True):
    """
    Crée un signal de test et affiche le spectre fréquentielle du signal avec et sans fenêtre de blackman.
    Peut afficher aussi le signal en entrée si plot_sinus est vrai

    :param amplitude: L'amplitude du signal qui sera utilisé pour le test
    :param frequency_test: La fréquence du signal de test
    :param fe: La fréquence d'échantillonnage
    :param N: le nombre d'échantillon dans le signal
    :param plot_sinus: True si on veux afficher le signal en entrée, mettre faux pour passer cette étape
    :return: Rien
    """
    print("Début du plan de test pour les points A")
    print("---------------------------------------")
    print("Point A1 : Aucun tests python")
    print("---------------------------------------")
    print("Point A2, A3, A4: Calcul du spectre d’entrée")
    print("Affichage du signal analyser")
    x_signal = create_sinus(amplitude, frequency_test, fe, N, plot_data=plot_sinus) #Création du sinus dans un format [data,fréquence]
    x_signal_window = x_signal[0] * np.blackman(len(x_signal[0]))
    x_signal_freq = np.arange(0, N)*fe/N
    x_signal_dft = np.fft.fft( x_signal[0])
    x_signal_window_dft = np.fft.fft(x_signal_window)

    plt.xlabel("Fréquence(Hz)")
    plt.ylabel("Amplitude(dB)")
    plt.title("Spectres des sinusoides de " + str(x_signal[1]) + "Hz avec et sans fenêtrage")
    plt.plot(x_signal_freq, 20*np.log10(np.abs(x_signal_dft)**2), label = "Spectre du sinus sans fenêtrage")
    plt.plot(x_signal_freq, 20 * np.log10(np.abs(x_signal_window_dft)**2), label="Spectre du sinus avec fenêtrage")
    plt.grid(which="both", axis="both")
    plt.legend()
    plt.show()
    print("---------------------------------------")
    print("---------------------------------------")

def plan_de_test_B(plot_filter = False):
    """
    Crée les filtres FIR passe-bas de 500Hz, passe-nade de 500 à 1500Hz, 1500 à 2500Hz, 2500Hz à 4500Hz et un passe-bas
    de 4490Hz. Affiche ensuite les spectres fréquentielles des filtres sur le même graphique. Il est possible d'afficher
    individuellement les filtre en changeant le paramètre plot_filter à true. Affiche, ensuite, le spectre de fréquence
    des filtres cummulatifs H7 + H6 + H5 + H4 + H3 et le filtre cummulatif H7 + H5 + H3. Finalement, il écrit les fichiers
    textes des coefficients de chacun des filtres dans des fichiers txt dans le répertoire du projet

    :param plot_filter: Détermine si les filtres seront affiché individuellement
    :return: Rien
    """
    print("Début du plan de test pour les points B")
    print("---------------------------------------")
    print("Point B: Affichages des filtres FIR décomposé et composé")
    [h_low_500, H_DFT_low_500] = create_filter(256, 500, 20000, 'blackman', 'lowpass',padding_zero=3*256, plot_data = plot_filter)
    [h_bp_500_1500, H_DFT_bp_500_1500] = create_filter(256, [500, 1500], 20000, 'blackman', 'bandpass',padding_zero=3*256, plot_data = plot_filter)
    [h_bp_1500_2500, H_DFT_bp_1500_2500] = create_filter(256, [1500, 2500], 20000, 'blackman', 'bandpass',padding_zero=3*256,  plot_data = plot_filter)
    [h_bp_2500_4500, H_DFT_bp_2500_4500] = create_filter(256, [ 2500, 4500], 20000, 'blackman', 'bandpass',padding_zero=3*256,  plot_data = plot_filter)
    [h_highpass_4490, H_DFT_highpass_4490] = create_filter(255, 4490, 20000, 'blackman', 'highpass',padding_zero=3*256 + 1,  plot_data = plot_filter)
    filter_list_DFT = [[H_DFT_low_500,"Passe-bas de 500Hz"] , [H_DFT_bp_500_1500,"Passe-bande de 500 à 1500Hz"],[H_DFT_bp_1500_2500, "Passe-bande de 1500 à 2500Hz"],
                       [H_DFT_bp_2500_4500, "Passe-bande de 2500 à 4500Hz"] ,[H_DFT_highpass_4490,"Passe-haut de 4490Hz"]]
    show_filter(filter_list_DFT, fe = 20000, show_H1_H3_H5=True)
    print("---------------------------------------")
    print("Point B: Écriture des fichiers text pour l'écriture du .h")
    make_array_C_Complex(H_DFT_low_500, title = "low pass 500")
    make_array_C_Complex(H_DFT_bp_500_1500, title = " band pass 500-1500")
    make_array_C_Complex(H_DFT_bp_1500_2500, title = " band pass 1500-2500")
    make_array_C_Complex(H_DFT_bp_2500_4500, title = " band pass 2500-4500")
    make_array_C_Complex(H_DFT_highpass_4490, title = "high pass 4500")
    print("Fin de l'écriture des fichier txt")
    print("---------------------------------------")
    print("Fin du point B")
    print("---------------------------------------")

def plan_de_test_C():
    """
    Création du filtre IIR de type Direct II en cascade avec le format Q2.13, Q2.5 et sans formatage. Il affiche
    ensuite les spectres de ces filtres sur un graphiques

    :return: Rien
    """
    [sos_Q213, sos_rounded_Q213, w_sos_rounded_Q213, h_dft_sos_rounded_Q213] = filter_IIR_ellip_SOS(filter_order=4, fc_low=950,
                                                                                fc_high=1050, pass_band_ripple_db=0.5,
                                                                                stop_band_attn_db=70,
                                                                                sampling_frequency=20000,
                                                                                filter_type="bandstop",
                                                                                Q_X=2, Q_Y=13, fe=20000.0,
                                                                                show_data=False)
    [sos_Q25, sos_rounded_Q25, w_sos_rounded_Q25, h_dft_sos_rounded_Q25] = filter_IIR_ellip_SOS(filter_order=4, fc_low=950,
                                                                                fc_high=1050, pass_band_ripple_db=0.5,
                                                                                stop_band_attn_db=70,
                                                                                sampling_frequency=20000,
                                                                                filter_type="bandstop",
                                                                                Q_X=2, Q_Y=5, fe=20000.0,
                                                                                show_data=False)

    [w_sos, h_dft_sos] = signal.sosfreqz(sos_Q213, worN=10000, fs=20000.0)
    plt.xlabel("Fréquence(Hz)")
    plt.ylabel("Amplitude(dB)")
    plt.title("Spectre d'amplitude du filtre IIR")
    plt.semilogx(w_sos_rounded_Q213, 20 * np.log10(np.abs(h_dft_sos_rounded_Q213)), label = "Filtre IIR Q2.13")
    plt.semilogx(w_sos_rounded_Q25, 20 * np.log10(np.abs(h_dft_sos_rounded_Q25)), label = "Filtre IIR Q2.5")
    plt.semilogx(w_sos, 20 * np.log10(np.abs(h_dft_sos)), label = "Filtre IIR format initiale")
    plt.grid(which="both", axis="both")
    plt.legend()
    plt.xlim(0,20000/2)
    plt.show()
    make_IIR_coefficient_file(sos_rounded_Q213, "Coefficient_SOS_ordre_4")

def optionnal_test():
    """
    Tests optionnelles pour valider le fonctionnement des filtres développé durant L'APP. Il test un filtre FIR et un filtre
    IIR avec plusieurs signaux. Pour changer le fonctionnement de cette fonction, il faut le changer dans le code directement

    :return: Rien
    """
    x_impulse = signal.unit_impulse(1000)
    x_250 = create_sinus(128, 250, 20000, 256)
    x_250_pad = [np.append(x_250[0], np.zeros(10000 - len(x_250[0]))), 250]
    x_750 = create_sinus(128, 750, 20000, 256)
    x_750_pad = [np.append(x_750[0], np.zeros(10000 - len(x_750[0]))), 750]
    x_1000 = create_sinus(128, 1000, 20000, 256)
    x_1000_pad = [np.append(x_1000[0], np.zeros(10000 - len(x_1000[0]))), 1000]
    x_1250 = create_sinus(128, 1250, 20000, 256)
    x_1250_pad = [np.append(x_1250[0], np.zeros(10000 - len(x_1250[0]))), 1250]
    x_1750 = create_sinus(128, 1750, 20000, 256)
    x_1750_pad = [np.append(x_1750[0], np.zeros(10000 - len(x_1250[0]))), 1250]
    x_2750 = create_sinus(128, 2750, 20000, 256)
    x_2750_pad = [np.append(x_2750[0], np.zeros(10000 - len(x_2750[0]))), 2750]
    x_3750 = create_sinus(128, 3750, 20000, 256)
    x_3750_pad = [np.append(x_3750[0], np.zeros(10000 - len(x_3750[0]))), 3750]
    x_5000 = create_sinus(128, 5000, 20000, 256)
    x_5000_pad = [np.append(x_5000[0], np.zeros(10000 - len(x_5000[0]))), 5000]

    [h_low_500, H_DFT_low_500] = create_filter(256, 500, 20000, 'blackman', 'lowpass',padding_zero=3*256, plot_data = False)
    [h_bp_500_1500, H_DFT_bp_500_1500] = create_filter(256, [500, 1500], 20000, 'blackman', 'bandpass',padding_zero=3*256, plot_data = False)
    [h_bp_1500_2500, H_DFT_bp_1500_2500] = create_filter(256, [1500, 2500], 20000, 'blackman', 'bandpass',padding_zero=3*256,  plot_data = False)
    [h_bp_2500_4500, H_DFT_bp_2500_4500] = create_filter(256, [ 2500, 4500], 20000, 'blackman', 'bandpass',padding_zero=3*256,  plot_data = False)
    [h_highpass_4490, H_DFT_highpass_4490] = create_filter(255, 4490, 20000, 'blackman', 'highpass',padding_zero=3*256 + 1,  plot_data = False)

    [sos, sos_rounded, w_sos_rounded, h_dft_sos_rounded] = filter_IIR_ellip_SOS(filter_order = 4, fc_low = 950, fc_high = 1050, pass_band_ripple_db = 1,
                                                                                stop_band_attn_db = 70, sampling_frequency = 20000 , filter_type = "bandstop",
                                                                                Q_X = 2, Q_Y = 13, fe = 20000.0, show_data = False)

    filter_list_DFT = [[H_DFT_low_500,"Passe-bas de 500Hz"] , [H_DFT_bp_500_1500,"Passe-bande de 500 à 1500Hz"],[H_DFT_bp_1500_2500, "Passe-bande de 1500 à 2500Hz"],
                       [H_DFT_bp_2500_4500, "Passe-bande de 2500 à 4500Hz"] ,[H_DFT_highpass_4490,"Passe-haut de 4490Hz"]]
    x_signals = [[x_impulse, 0], x_250, x_750, x_1000, x_1250, x_1750, x_2750, x_3750, x_5000]
    x_signals_pad = [x_250_pad, x_750_pad, x_1000_pad, x_1250_pad, x_2750_pad, x_3750_pad, x_5000_pad]

    #Vous pourvez sélectionner le filtre à tester ici:
    test_filter_convolution(h_bp_1500_2500, x_signals)
    test_filter_SOS(sos_rounded/(2**13), x_signals)




#****************************************************************************************************************************
#Création des données/filtres
#****************************************************************************************************************************
def create_sinus(amplitude, frequency, fe, number_of_sampling, plot_data = False):
    """
    Crée un sinus avec les paramètre mis en entrée

    :param amplitude: Amplitude du sinus
    :param frequency: Fréquence du Sinus
    :param fe: Fréquence d'échantillonnage du sinus
    :param number_of_sampling: La quantité de données du sinus
    :param plot_data: Affiche le sinus si ce paramètre est vrai, sinon aucun affichage
    :return: Sinusoide avec les paramètres mis en entrée
    """
    n = np.arange(0, number_of_sampling)
    sinus = amplitude*np.sin(2*np.pi*frequency*n/fe)
    if plot_data == True:
        print("Le nombre d'échantillon par période est : " + str(fe/frequency) )
        plt.title("Valeur du sinus à une fréquence de " + str(frequency) + "Hz Avec une échantillon de " + str(fe) + "Hz")
        plt.xlabel("échantillon [n]")
        plt.ylabel("amplitude [V]")
        plt.xlim(0, 10*(fe/frequency))
        plt.plot(sinus)
        plt.show()
    return [sinus, frequency]

def create_filter(filter_order, cutoff_fc, sampling_freq, window_type = 'balckman', filter_type = 'lowpass', padding_zero = 0, plot_data = False):
    """
    Crée un filtre de type FIR avec les paramètres mis en entrée

    :param filter_order: L'ordre du filtre
    :param cutoff_fc: La fréquence de coupure du filtre
    :param sampling_freq: la fréquence d'échantillonnage
    :param window_type: Le type de fenêtre appliqué au filtre
    :param filter_type: Le type de filtre (ex. lowpass)
    :param padding_zero: Ajoute des 0  lors de la création du FIR . Par exemple, si le filtre est d'ordre N, on veut avoir 4N,
                        il Faut mettre 3N au padding (N + 3N = 4N)
    :param plot_data: Affiche le spectre du filtre si ce paramétre est vrai
    :return: [h, H_DFT] : h = Réponse impulsionnelle du filtre
                          H_DFT = Le spectre dans el domaine fréquentielle du filtre
    """
    h = signal.firwin(numtaps =  filter_order, cutoff = cutoff_fc, window= window_type, pass_zero=filter_type, fs = sampling_freq)
    if plot_data == True:
        plt.plot(h)
        plt.show()
        print(len(h))
    H_DFT = np.fft.fft(h, len(h) + padding_zero)
    if plot_data == True:
        plt.title("Valeur Filtre sans remise en f")
        plt.semilogx(20*np.log10(np.abs(H_DFT)))
        plt.show()
        print(len(H_DFT))
    f = np.arange(0, len(H_DFT)) * ( sampling_freq/ len(H_DFT))
    if plot_data == True:
        plt.title("Valeur Filtre avec remise en f")
        plt.plot(f, 20 * np.log10(np.abs(H_DFT)))
        plt.show()
    return [h, H_DFT]

def filter_IIR_ellip_SOS(filter_order, fc_low, fc_high, pass_band_ripple_db, stop_band_attn_db, sampling_frequency, filter_type, Q_X, Q_Y, fe : float, show_data = False):
    """
      Crée un filtre de type FIR de type elliptique avec les paramètres mis en entrée. Pour le moment, il ne fonctionne seulement que pour les filtre passe-bande/coupe-bande. Il faudrait modifier
      la fonction pour qu'elle fonctionne avec les passe-haut et passe-bas.

    :param filter_order: L'ordre du filtre
    :param fc_low: Fréquence de coupure basse
    :param fc_high: Fréquence de coupure haute
    :param pass_band_ripple_db: Le ripple maximale de la bande passante
    :param stop_band_attn_db: L'atténuation minimale dans la bande de coupure
    :param sampling_frequency: La fréquence d'échantillonnage
    :param filter_type: Le type de filtre
    :param Q_X: Le formatage des donnée (X)
    :param Q_Y: Le formatage des donnée (Y)
    :param fe: Fréquence d'échantillonnage
    :param show_data: Affiche le spectre du filtre, l'amplitude des pôles et des zéros dans le terminale si ce paramétre est vrai
    :return: [sos, sos_rounded, w_sos_rounded, h_dft_sos_rounded]
            - sos : les coefficients des filtres d'ordre 2 sans formattage
            - sos_rounded : les coefficients des filtres d'ordre 2 avec le formatage
            - w_sos_rounded : Les fréquences dont le dft a été calculé
            -h_dft_sos_rounded: le spectre en fréquence du filtre IIR avec le formattage
    """
    sos = signal.ellip(N=filter_order, rp=pass_band_ripple_db, rs=stop_band_attn_db, Wn=[fc_low, fc_high], fs=sampling_frequency
                       ,btype=filter_type, output="sos")
    sos_rounded = np.round(sos*np.power(Q_X,Q_Y))
    [w_sos_rounded, h_dft_sos_rounded] = signal.sosfreqz(sos_rounded/np.power(Q_X,Q_Y), worN=10000, fs=fe)
    if show_data == True:
        [z, p, k] = signal.sos2zpk(sos_rounded / np.power(Q_X, Q_Y))
        print(np.abs(z))
        print(np.abs(p))
        plt.semilogx(w_sos_rounded, 20 * np.log10(np.abs(h_dft_sos_rounded)))
        plt.show()
    return [sos, sos_rounded, w_sos_rounded, h_dft_sos_rounded]



#----------------------------------------------------------------------------------------------------
#Fonction de tests optionnelles
#----------------------------------------------------------------------------------------------------
def test_filter_convolution(filter_to_test, list_of_signal):
    """
    Fonction de tests optionnelles pour tester les filtres par convolution

    :param filter_to_test: réponse impulsionnelle du filtre a tester
    :param list_of_signal: Liste des signaux a tester. Les signaux doit être dans le format [[données_du_signal], fréquence_du_signal]
    :return: Rien
    """
    nb_signals = len(list_of_signal)
    index_signal = 0
    fig = plt.figure()
    for signal in list_of_signal:
        convo = np.convolve(filter_to_test, signal[0])
        ax1 = fig.add_subplot((nb_signals*100) + 10 + (index_signal + 1))
        ax1.plot(convo)
        ax1.set_title("signal at a frequency of : " + str(signal[1]))
        index_signal+=1
    plt.show()


def test_filter_SOS(sos_input, signal_input):
    """
    Fonction de tests optionnelles pour tester les filtres dans le format SOS

    :param filter_to_test: réponse impulsionnelle du filtre a tester
    :param list_of_signal: Liste des signaux a tester. Les signaux doit être dans le format [[données_du_signal], fréquence_du_signal]
    :return: Rien
    """
    nb_signals = len(signal_input)
    index_signal = 0
    fig = plt.figure()
    for x in signal_input:
        result = signal.sosfilt(sos_input, x[0])
        ax1 = fig.add_subplot((nb_signals*100) + 10 + (index_signal + 1))
        ax1.plot(result)
        ax1.set_title("signal at a frequency of : " + str(x[1]))
        index_signal+=1
    plt.show()



#----------------------------------------------------------------------------------------------------
#Affichage des données
#----------------------------------------------------------------------------------------------------
def show_filter(filter_list, fe = 20000, show_H1_H3_H5 = True):
    """
    Affichage des filtres dans la liste de filtre et affichage du filtre cummulatif.
    Permet d'afficher le filtre cummulatif H7_H5_H3 si show_H1_H3_H5 est vrai

    :param filter_list: La liste des spectres dans le domaine fréquentiel des filtres. Les filtre
                        doivent être dans le format [DFT, nom_du_filtre]
    :param fe: La fréquence d'échantillonnage
    :param show_H1_H3_H5: afficher le filtre cummulatif H7_H5_H3 si show_H1_H3_H5 est vrai
    :return: Rien
    """
    H_tot = 0
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.grid(which = "both", axis = "both")
    plt.title("Fonctions de transfert Hx[f]")
    freq = np.arange(0, len(filter_list[0][0]))*fe/len(filter_list[0][0])
    for filter_DFT in filter_list:
        ax1.semilogx(freq, 20*np.log10(np.abs(filter_DFT[0])), label = filter_DFT[1])
        H_tot += filter_DFT[0]
    ax1.set_xlim([0, int(fe / 2)])
    ax1.set_ylim([-40, 10])
    plt.legend()
    plt.ylabel("Gain [dB]")
    plt.xlabel("Fréquence [Hz]")

    ax2 = fig.add_subplot(212)
    plt.grid(which="both", axis="both")
    plt.title("Fonction de transfert cummulatif H7 + H6 + H5 + H4 + H3")
    plt.ylabel("Gain [dB]")
    plt.xlabel("Fréquence [Hz]")
    ax2.semilogx(freq, 20*np.log10(np.abs(H_tot)))
    ax2.set_xlim([0, int(fe / 2)])
    ax2.set_ylim([-40, 10])
    plt.show()

    if show_H1_H3_H5 == True:
        print("Affichage optionnel du filtre composé de H1, H3, H5")
        H1_H3_H5 = filter_list[0][0] + filter_list[2][0] + filter_list[4][0]
        plt.title("Fonction de transfert cummulatif H7  + H5  + H3")
        plt.ylabel("Gain [dB]")
        plt.xlabel("Fréquence [Hz]")
        plt.plot(freq, 20 * np.log10(np.abs(H1_H3_H5)))
        plt.grid(which="both", axis="both")
        plt.xlim(0, int(fe / 2))
        plt.show()




#----------------------------------------------------------------------------------------------------
#Création des fichiers
#----------------------------------------------------------------------------------------------------

def make_array_C(array_to_convert, title = ""):
    """
    Création et écriture des données dans un array dans un fichier txt en format Q2.13 afin
    d'être utilisé dans un fichier .h ou .c
    :param array_to_convert: L'array a écrire
    :param title: Le titre du fichier
    :return: Rien
    """
    try:
        f = open(title+".txt", "w", encoding="utf-8")
        f.write(title + "\n")
        f.write("Length of the array = " + str(len(array_to_convert)) + "\n")
        for element in array_to_convert:
            for element in array_to_convert:
                print(str(int(np.round(element * 8192))) + ",\n")
        f.write("End of the array ------\n")
    except ValueError:
        print("error")
        f.close()
    except OSError:
        print("error")
        f.close()
def make_array_C_Complex(array_to_convert, title = ""):
    """
    Création et écriture des données de type complexe dans un array dans un fichier txt en format Q2.13 afin
    d'être utilisé dans un fichier .h ou .c
    :param array_to_convert: L'array a écrire
    :param title: Le titre du fichier
    :return: Rien
    """
    try:
        f = open(title+".txt", "w", encoding="utf-8")
        f.write(title + "\n")
        f.write("Length of the array = " + str(len(array_to_convert)) + "\n\n")
        f.write("{")
        for element in array_to_convert:
            element = element * 8192
            real_str = int(np.round(element.real))
            imag_str = int(np.round(element.imag))
            f.write("{" + str(real_str) + " , " + str(imag_str) + "},\n")
    except ValueError:
        f.close()
    except OSError:
        f.close()
    f.write("}")
    f.close()

def make_IIR_coefficient_file(SOS_to_convert, title = ""):
    """
    Création et écriture des coefficient pour un filtre IIR de type SOS dans un fichier txt afin
    d'être utilisé dans un fichier .h ou .c.

    Cette fonction ne convertie par les données en format QXY
    :param array_to_convert: Les données a écrire
    :param title: Le titre du fichier
    :return: Rien
    """
    try:
        f = open(title+".txt", "w", encoding="utf-8")
        f.write(title + "\n")
        f.write("{")
        for coefficients in SOS_to_convert:
            f.write("{ ")
            for coefficient in coefficients:
                f.write(str(int(coefficient)) + " , ")
            f.write("},\n")
        f.write("}")
    except ValueError:
        f.close()
    except OSError:
        f.close()
    f.close()



def main():
    plan_de_test_A(amplitude = 1, frequency_test  = 1000, fe = 20000, N = 768, plot_sinus = True)
    plan_de_test_B()
    plan_de_test_C()
    #optionnal_test() # Tests optionnelles afin de tester les fréquences.

if __name__ == "__main__":
    main()



