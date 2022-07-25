import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from scipy import signal


def plot_complex_csv_data(file_name: str, scale_factor: float = 1, fe: float = 20000):
    """
    Plot the norm in dB of a complex array loaded from a .csv file

    :param file_name: name of .csv file containing complex array in second column,
                      with interleaved real/imaginary hexadecimal components
    :param scale_factor: scale factor, optional (default = 1, i.e. no scaling)
                         ex: scale_factor = 2**Y, to remove scaling in QXY encoding of H
    :param fe: sampling frequency in Hz, optional (default = 20 kHz)
    :return: None
    """

    # Load re/im interleaved hex data from .csv file as unsigned 64 bit integer array
    # NB: int(s, 0) converts a string containing a hexadecimal number with leading "0x"
    #     to an unsigned integer, but np.int64() is required as a wrapper otherwise
    #     genfromtxt() tries to load result into a 32 bit integer and overflow may occur
    data: np.ndarray = np.genfromtxt(
        file_name,
        delimiter=",",
        usecols=[1],
        converters={1: lambda s: np.int64(int(s, 0))},
    )
    data[data >= 0x80000000] -= 0x100000000  # Convert to signed 64 bit integers
    x: np.ndarray = data[0::2] + 1j * data[1::2]  # Convert to complex array of floats

    # Build frequency array for plotting
    f: np.ndarray = np.arange(0, fe, fe / len(x))

    # Calculate vector norm, replace any null values with array minimum (non-zero)
    # so log10() function doesn't complain about null values, convert to dB
    x_norm: np.ndarray = (np.abs(x)**2)/scale_factor
    x_norm[x_norm == 0] = np.amin(x_norm[x_norm > 0])
    x_norm_db: np.ndarray = 10 * np.log10(x_norm+1)

    # Plot norm in dB as a function of frequency, on linear (like DMCI) & log scales
    fig, axs = plt.subplots(3)
    fig.suptitle(f"{file_name} (fe = {fe} Hz, scale factor = {scale_factor})")
    axs[0].plot(f, x_norm_db)
    axs[0].set_xlabel("FrÃ©quence sur Ã©chelle linÃ©aire [0, fe] (Hz)")
    axs[0].set_ylabel("Amplitude [dB]")
    axs[0].grid(which="both")
    axs[1].plot(
        np.concatenate((f[len(x) // 2 :] - fe, f[: len(x) // 2])),
        np.concatenate((x_norm_db[len(x) // 2 :], x_norm_db[: len(x) // 2])),
    )
    axs[1].set_xlabel("FrÃ©quence sur Ã©chelle linÃ©aire [-fe/2, fe/2] (Hz)")
    axs[1].set_ylabel("Amplitude [dB]")
    axs[1].grid(which="both")
    axs[2].semilogx(f[: len(x) // 2], x_norm_db[: len(x) // 2])
    axs[2].set_xlabel("FrÃ©quence sur Ã©chelle logarithmique [0, fe/2] (Hz)")
    axs[2].set_ylabel("Amplitude [dB]")
    axs[2].grid(which="both")
    plt.tight_layout()
    plt.show()

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

def make_array_C(array_to_convert, title = ""):
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

    try:
        f = open(title+".txt", "w", encoding="utf-8")
        f.write(title + "\n")
        f.write("Length of the array = " + str(len(array_to_convert)) + "\n")
        for element in array_to_convert:
            element = element * 8192
            real_str = int(np.round(element.real))
            imag_str = int(np.round(element.imag))
            f.write("{" + str(real_str) + " , " + str(imag_str) + "},\n")
    except ValueError:
        f.close()
    except OSError:
        f.close()
    f.close()

def show_filter(filter_list):
    plt.title("filter result: ")
    H_tot = 0
    plt.ylim([-40,10])
    for filter_DFT in filter_list:
        plt.semilogx(20*np.log10(np.abs(filter_DFT)))
        H_tot += filter_DFT*8000
    plt.show()
    plt.plot(20*np.log10(np.abs(H_tot)))
    plt.show()


def test_B2(x_signal, h_filter_DFT):
    x_signal_DFT = np.fft.fft(x_signal)
    compteur = 0
    y = list()
    for element_X in x_signal_DFT:
        a = element_X.real
        b = element_X.imag
        c = h_filter_DFT[compteur].real
        d = h_filter_DFT[compteur].imag
        y.append(np.complex((a*c)-(b*d), -1*((a*d)+(b*c))))
    print(y)
    y_ifft = np.fft.fft(y)
    for element_y in y:
        a = element_y.real
        b = element_y.imag
        y.append(np.complex((a*c)-(b*d), -1*((a*d)+(b*c))))











def filter_IIR_ellip_SOS(filter_order, fc_low, fc_high, pass_band_ripple_db, stop_band_attn_db, sampling_frequency, filter_type, Q_X, Q_Y, fe : float, show_data = False):
    sos = signal.ellip(N=filter_order, rp=pass_band_ripple_db, rs=stop_band_attn_db, Wn=[fc_low, fc_high], fs=sampling_frequency
                       ,btype=filter_type, output="sos")

    print(sos)


    sos_rounded = np.round(sos*np.power(2,13))
    [z, p, k] = signal.sos2zpk(sos_rounded / np.power(2, 13))
    print(np.abs(z))
    print(np.abs(p))
    [w_sos_rounded, h_dft_sos_rounded] = signal.sosfreqz(sos_rounded/np.power(2,13), worN=10000, fs=fe)
    if show_data == True:
        plt.semilogx(w_sos_rounded, 20 * np.log10(np.abs(h_dft_sos_rounded)))
        plt.show()
    return [sos, sos_rounded, w_sos_rounded, h_dft_sos_rounded]

def make_IIR_coefficient_file(SOS_to_convert, title = ""):
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











[h_low_500, H_DFT_low_500] = create_filter(256, 500, 20000, 'blackman', 'lowpass',padding_zero=3*256, plot_data = False)
[h_bp_500_1500, H_DFT_bp_500_1500] = create_filter(256, [500, 1500], 20000, 'blackman', 'bandpass',padding_zero=3*256, plot_data = False)
[h_bp_1500_2500, H_DFT_bp_1500_2500] = create_filter(256, [1500, 2500], 20000, 'blackman', 'bandpass',padding_zero=3*256,  plot_data = False)
[h_bp_2500_4500, H_DFT_bp_2500_4500] = create_filter(256, [ 2500, 4500], 20000, 'blackman', 'bandpass',padding_zero=3*256,  plot_data = False)
[h_highpass_4490, H_DFT_highpass_4490] = create_filter(255, 4490, 20000, 'blackman', 'highpass',padding_zero=3*256 + 1,  plot_data = True)


filter_list_DFT = [H_DFT_low_500, H_DFT_bp_500_1500,H_DFT_bp_1500_2500, H_DFT_bp_2500_4500 , H_DFT_highpass_4490]
show_filter(filter_list_DFT)

#make_array_C_Complex(H_DFT_low_500, title = "low pass 500")
make_array_C_Complex(H_DFT_bp_500_1500, title = " band pass 500-1500")
make_array_C_Complex(H_DFT_bp_1500_2500, title = " band pass 1500-2500")
make_array_C_Complex(H_DFT_bp_2500_4500, title = " band pass 2500-4500")
make_array_C_Complex(H_DFT_highpass_4490, title = "high pass 4500")



x_impulse = signal.unit_impulse(1000)
x_250 = create_sinus(128, 250,20000,256)
x_250_pad = [np.append(x_250[0], np.zeros(10000-len(x_250[0]))),250]
x_750 = create_sinus(128,750,20000,256)
x_750_pad = [np.append(x_750[0], np.zeros(10000-len(x_750[0]))),750]
x_1000 = create_sinus(128, 1000, 20000,256)
x_1000_pad = [np.append(x_1000[0], np.zeros(10000-len(x_1000[0]))),1000]
x_1250 = create_sinus(128,1250,20000,256)
x_1250_pad = [np.append(x_1250[0], np.zeros(10000-len(x_1250[0]))),1250]
x_1750 = create_sinus(128,1750,20000,256)
x_1750_pad = [np.append(x_1750[0], np.zeros(10000-len(x_1250[0]))),1250]
x_2750 = create_sinus(128,2750,20000,256)
x_2750_pad = [np.append(x_2750[0], np.zeros(10000-len(x_2750[0]))),2750]
x_3750 = create_sinus(128,3750,20000,256)
x_3750_pad = [np.append(x_3750[0], np.zeros(10000-len(x_3750[0]))),3750]
x_5000 = create_sinus(128, 5000,20000,256)
x_5000_pad = [np.append(x_5000[0], np.zeros(10000-len(x_5000[0]))), 5000]





x_signals = [[x_impulse, 0], x_250, x_750, x_1000, x_1250, x_1750, x_2750, x_3750, x_5000]
x_signals_pad = [x_250_pad, x_750_pad, x_1000_pad, x_1250_pad, x_2750_pad, x_3750_pad, x_5000_pad]


[sos, sos_rounded, w_sos_rounded, h_dft_sos_rounded] = filter_IIR_ellip_SOS(filter_order = 4, fc_low = 950, fc_high = 1050, pass_band_ripple_db = 1,
                                                                            stop_band_attn_db = 70, sampling_frequency = 20000 , filter_type = "bandstop",
                                                                            Q_X = 2, Q_Y = 13, fe = 20000.0, show_data = True)

make_IIR_coefficient_file(sos_rounded, "Coefficient_SOS_ordre_4")
print(sos_rounded)
#test_filter_convolution(h_bp_1500_2500, x_signals)
test_filter_SOS(sos_rounded/(2**13), x_signals)
#plot_complex_csv_data("outFFT.csv", 2**31,  20000)

x_2000 = create_sinus(128, 2000,20000,1024, plot_data=False)
plt.title("DFT du sinus à 2kHz avec et sans fenêtre de blackman")
plt.xlabel("Fréquences [Hz]")
plt.ylabel("Amplitude [dB]")
x_window_blackman = x_2000[0]*np.blackman(len(x_2000[0]))
x_window_blackman_DFT = np.fft.fft(x_window_blackman)
x_2000_DFT = np.fft.fft(x_2000[0])
plt.plot(20*np.log10(np.abs(x_2000_DFT)), label = "DFT du sinus sans fenêtrage")
plt.plot(20*np.log10(np.abs(x_window_blackman_DFT)), label = "DFT du sinus avec fenêtrage")
plt.legend()
plt.show()



