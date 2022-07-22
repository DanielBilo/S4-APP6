import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from zplane import zplane


def probleme_1(fe: float):
    """
    ProblÃ¨me 1: Filtre IIR elliptique
    """

    # Filter specifications
    fc_low: float = 900
    fc_high: float = 1100
    filter_order: int = 2
    pass_band_ripple_db: float = 1
    stop_band_attn_db: float = 40

    # Filter coefficients
    [b, a] = signal.ellip(
        N=filter_order,
        rp=pass_band_ripple_db,
        rs=stop_band_attn_db,
        Wn=[fc_low, fc_high],
        fs=fe,
        btype="bandpass",
        output="ba",
    )

    sos = signal.ellip(
        N=filter_order,
        rp=pass_band_ripple_db,
        rs=stop_band_attn_db,
        Wn=[fc_low, fc_high],
        fs=fe,
        btype="bandpass",
        output="sos",
    )




    f1 = 200
    f2 = 2000
    t = np.arange(0, 4*512)
    print(t)
    sin1 = np.sin(2 * np.pi * f1 / 20000 * t)
    sin2 = np.sin(2 * np.pi * f2 / 20000 * t)
    plt.figure()
    plt.plot(sin1)
    plt.show()

    plt.figure()
    plt.plot(sin2)
    plt.show()

    SIN1 = 20 * np.log10(np.abs(np.fft.fft(sin1)))
    FFTsin1 = (np.fft.fft(sin1))
    SIN2 = 20 * np.log10(np.abs(np.fft.fft(sin2)))
    FFTsin2 = (np.fft.fft(sin2))
    # plt.figure()
    # plt.xlim(0, 1024)
    # plt.plot(SIN1)
    # plt.show()
    #
    # plt.figure()
    # plt.xlim(0, 1024)
    # plt.plot(SIN2)
    # plt.show()

    # scipy.signal.firwin(numtaps, cutoff, width=None, window='hamming', pass_zero=True, scale=True, nyq=None, fs=None)
    FIR_low = signal.firwin(512, 1000, width=None, window='hamming', pass_zero='lowpass', scale=True, nyq=None, fs=20000)
    FIR_low_pad = np.append(FIR_low, np.zeros(3 * 512))
    FFTLP = (np.abs(np.fft.fft(FIR_low_pad)))

    # plt.figure()
    # plt.plot(FIR_low)
    # plt.show()

    FIR_high = signal.firwin(513, 950, width=None, window='hamming', pass_zero='highpass', scale=True, nyq=None, fs=20000)
    FIR_high_pad = np.append(FIR_high, np.zeros(3 * 513))
    FFTHP = (np.abs(np.fft.fft(FIR_high_pad)))
    # plt.figure()
    # plt.xlim(0, 1024)
    # plt.plot(20 * np.log10(FFTHP))
    # plt.plot(20 * np.log10(FFTLP))
    # plt.show()
    #
    # plt.figure()
    # plt.xlim(0, 1024)
    # plt.plot(FFTHP[0:2048] + FFTLP) #addioner pas en dB pour obtenir la ligne flat
    # plt.show()

    # zplane(b, a) #schémas des zéros
    zeros = np.roots(b) #num
    poles = np.roots(a) #denum
    x = signal.unit_impulse(1000)

    # signal.unit_impulse
    y = signal.lfilter(b, a,x)
    # print(y)
    # plt.figure()
    # plt.plot(y)
    # plt.show()

    # Frequency response
    bQ213 = np.round(b * np.power(2, 13)) / np.power(2,13)
    aQ213 = np.round(a * np.power(2, 13)) / np.power(2,13)
    sosQ213 = np.round(sos * np.power(2, 13)) / (np.power(2, 13))

    [w, h_dft] = signal.freqz(b, a, worN=10000, fs=fe)
    [w2, h_dft2] = signal.sosfreqz(sosQ213, worN=10000, fs=fe)
    [w3, h_dft3] = signal.freqz(bQ213, aQ213, worN=10000, fs=fe)

    sin1filtr = FFTsin1 * FFTLP[0: 2048]
    # sin1filtr = sin1filtr * FFTLP
    sin1filtr = np.fft.ifft(sin1filtr)

    FFTHP = (np.fft.fft(FIR_high_pad))
    sin2filtr = FFTsin2 * FFTHP[0: 2048]
    # sin2filtr = sin2filtr * FFTLP
    sin2filtr = np.fft.ifft(sin2filtr)


    plt.figure()
    # plt.xlim(0, 1024)
    plt.plot(np.real(sin1filtr[0:500]))
    plt.show()

    plt.figure()
    plt.xlim(0, 512)
    plt.title('signal filtré')
    plt.plot(np.real(sin2filtr))
    plt.show()




    # plt.figure()
    # plt.semilogx(w, 20 * np.log10(np.abs(h_dft)))
    # plt.semilogx(w2, 20 * np.log10(np.abs(h_dft2)))
    # plt.semilogx(w3, 20 * np.log10(np.abs(h_dft3)))
    # plt.title(f"RÃ©ponse en frÃ©quence du filtre elliptique (ordre {filter_order})")
    # plt.xlabel("FrÃ©quence [Hz]")
    # plt.ylabel("Gain [dB]")
    # plt.grid(which="both", axis="both")
    # plt.tight_layout()




def laboratoire():
    plt.ion()  # Comment out if using scientific mode!

    fe = 20000
    probleme_1(fe)
    print("Done!")


if __name__ == "__main__":
    laboratoire()