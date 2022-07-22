import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from scipy import signal

#****************************************************************************************************************************
#Problématique
#****************************************************************************************************************************
def create_sinus(amplitude, frequency, fe, number_of_sampling, plot_data = False):
    n = np.arange(0, number_of_sampling)
    sinus = amplitude*np.sin(2*np.pi*frequency*n/fe)
    if plot_data == True:
        print(fe/frequency )
        plt.title("Valeur du sinus à une fréquence de " + str(frequency) + "Hz Avec une échantillon de " + str(fe))
        plt.plot(sinus)
        plt.show()
    return [sinus, frequency]

def create_filter(filter_order, cutoff_fc, sampling_freq, window_type = 'balckman', filter_type = 'lowpass', padding_zero = 0, plot_data = False):
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

def test_filter_convolution(filter_to_test, list_of_signal):
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

def test_filter_frequential(filter_to_test_DFT, list_of_signal_DFT):
    nb_signals = len(list_of_signal_DFT)
    index_signal = 0
    fig = plt.figure()
    for signal in list_of_signal_DFT:
        result = filter_to_test_DFT * signal[0]
        ax1 = fig.add_subplot((nb_signals*100) + 10 + (index_signal + 1))
        ax1.plot(20*np.log10(result))
        ax1.set_title("signal at a frequency of : " + str(signal[1]))
        index_signal+=1
    plt.show()

def test_filter_SOS(sos_input, signal_input):
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

def filter_IIR_ellip_SOS(filter_order, fc_low, fc_high, pass_band_ripple_db, stop_band_attn_db, sampling_frequency, filter_type, Q_X, Q_Y, fe : float, show_data = False):
    sos = signal.ellip(N=filter_order, rp=pass_band_ripple_db, rs=stop_band_attn_db, Wn=[fc_low, fc_high], fs=sampling_frequency
                       ,btype=filter_type, output="sos")
    #Arrondissement des SOS
    sos_rounded = np.round(sos*np.power(2,13))
    [w_sos_rounded, h_dft_sos_rounded] = signal.sosfreqz(sos_rounded/np.power(2,13), worN=10000, fs=fe)
    if show_data == True:
        plt.semilogx(w_sos_rounded, 20 * np.log10(np.abs(h_dft_sos_rounded)))
        plt.show()
    return [sos, sos_rounded, w_sos_rounded, h_dft_sos_rounded]






[h_low_500, H_DFT_low_500] = create_filter(256, 500, 20000, 'blackman', 'lowpass', plot_data = False)
[h_bp_500_1500, H_DFT_bp_500_1500] = create_filter(256, [500, 1500], 20000, 'blackman', 'bandpass', plot_data = False)
[h_bp_1500_2500, H_DFT_bp_1500_2500] = create_filter(256, [1500, 2500], 20000, 'blackman', 'bandpass', plot_data = False)
[h_bp_2500_3500, H_DFT_bp_2500_3500] = create_filter(256, [ 2500, 3500], 20000, 'blackman', 'bandpass', plot_data = False)
[h_highpass_4490, H_DFT_highpass_4490] = create_filter(255, 4490, 20000, 'blackman', 'highpass', plot_data = False)

x_250 = create_sinus(128, 250,20000,256)
x_250_pad = [np.append(x_250[0], np.zeros(10000-len(x_250[0]))),250]
x_750 = create_sinus(128,750,20000,256)
x_750_pad = [np.append(x_750[0], np.zeros(10000-len(x_750[0]))),750]
x_1000 = create_sinus(128, 1000, 20000,256)
x_1000_pad = [np.append(x_1000[0], np.zeros(10000-len(x_1000[0]))),1000]
x_1250 = create_sinus(128,1250,20000,256)
x_1250_pad = [np.append(x_1250[0], np.zeros(10000-len(x_1250[0]))),1250]
x_2750 = create_sinus(128,2750,20000,256)
x_2750_pad = [np.append(x_2750[0], np.zeros(10000-len(x_2750[0]))),2750]
x_3750 = create_sinus(128,3750,20000,256)
x_3750_pad = [np.append(x_3750[0], np.zeros(10000-len(x_3750[0]))),3750]
x_5000 = create_sinus(128, 5000,20000,256)
x_5000_pad = [np.append(x_5000[0], np.zeros(10000-len(x_5000[0]))), 5000]


x_signals = [x_250, x_750, x_1000, x_1250, x_2750, x_3750, x_5000]
x_signals_pad = [x_250_pad, x_750_pad, x_1000_pad, x_1250_pad, x_2750_pad, x_3750_pad, x_5000_pad]


[sos, sos_rounded, w_sos_rounded, h_dft_sos_rounded] = filter_IIR_ellip_SOS(filter_order = 4, fc_low = 950, fc_high = 1050, pass_band_ripple_db = 0.5,
                                                                            stop_band_attn_db = 70, sampling_frequency = 20000 , filter_type = "bandpass",
                                                                            Q_X = 2, Q_Y = 13, fe = 20000.0, show_data = False)

test_filter_SOS(sos, x_signals)

