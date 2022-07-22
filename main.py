#
# Copyright (c) 2011 Christopher Felton
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# The following is derived from the slides presented by
# Alexander Kain for CS506/606 "Special Topics: Speech Signal Processing"
# CSLU / OHSU, Spring Term 2011.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from scipy import signal

def filter_IIR_ellip(filter_order, fc_low, fc_high, pass_band_ripple_db, stop_band_attn_db, fe : float):
    [b,a] = signal.ellip(N = filter_order,rp = pass_band_ripple_db, rs = stop_band_attn_db, Wn = [fc_low, fc_high], fs = fe, btype="bandpass", output = "ba" )
    sos = signal.ellip(N=filter_order, rp=pass_band_ripple_db, rs=stop_band_attn_db, Wn=[fc_low, fc_high], fs=fe,
                          btype="bandpass", output="sos")
    #Arrondissement des SOS
    sos_rounded = np.round(sos*np.power(2,13))
    b_rounded = np.round(b*np.power(2,13))
    a_rounded = np.round(a*np.power(2,13))

    [w_ab, h_dft_ab] = signal.freqz(b,a,worN = 10000, fs = fe)
    [w_sos, h_dft_sos] = signal.sosfreqz(sos, worN=10000, fs=fe)
    [w_ab_rounded, h_dft_ab_rounded] = signal.freqz(b_rounded,a_rounded,worN = 10000, fs = fe)
    [w_sos_rounded, h_dft_sos_rounded] = signal.sosfreqz(sos_rounded/np.power(2,13), worN=10000, fs=fe)

    plt.figure()
    plt.semilogx(w_ab, 20*np.log10(np.abs(h_dft_ab)))
    plt.semilogx(w_sos, 20 * np.log10(np.abs(h_dft_sos)))
    plt.semilogx(w_ab_rounded, 20 * np.log10(np.abs(h_dft_ab_rounded)))
    plt.semilogx(w_sos_rounded, 20 * np.log10(np.abs(h_dft_sos_rounded)))
    plt.show()

    return [b,a, w_ab, h_dft_ab]

def impulse_respond(b, a):
    h = signal.unit_impulse(1000)
    impuls_resp = signal.lfilter(b,a,h)
    plt.plot(impuls_resp)
    plt.show()

def FIR_overlap_save():
    low_pass = signal.firwin(512, cutoff = 1000, fs = 20000)
    high_pass = signal.firwin(513, cutoff = 950, pass_zero= "highpass", fs = 20000)
    plt.plot(low_pass)
    plt.show()
    plt.plot(high_pass)
    plt.show()


    H_fft_Low = np.fft.fft(low_pass, 4*513)
    H_FFT_High = np.fft.fft(high_pass, 4*513)
    f_low = np.arange(0, len(H_fft_Low))*((4*513)/len(low_pass))
    f_high = np.arange(0, len(H_FFT_High)) * ((4*513)/len(high_pass))
    plt.semilogx(f_low, 20 * np.log10(np.abs(H_fft_Low)))
    plt.semilogx(f_high, 20 * np.log10(np.abs(H_FFT_High)))
    plt.show()
    plt.semilogx(f_high, (20 * np.log10(np.abs(H_fft_Low + H_FFT_High))))
    plt.show()

    return [low_pass, high_pass, H_fft_Low, H_FFT_High]



def create_sinus():
    n = np.arange(0, 4*513)
    x_200 = np.sin(2*np.pi*200*n/20000)
    x_2000 = np.sin(2 * np.pi * 2000 *n/ 20000)
    x_200_fft = np.fft.fft(x_200)
    x_2000_fft = np.fft.fft(x_2000)
    plt.plot(x_200)
    plt.show()
    plt.plot(x_2000)
    plt.show()

    [low_pass, high_pass, H_FFT_LOW, H_FFT_HIGH] = FIR_overlap_save()
    x_200_filtered_low = x_200_fft * H_FFT_LOW
    x_2000_filtered_low = x_2000_fft *H_FFT_LOW
    x_200_filtered_High = x_200_fft * H_FFT_HIGH
    x_2000_filtered_High = x_2000_fft *H_FFT_HIGH

    plt.title("Sinus poop")
    plt.plot(np.fft.ifft(x_200_filtered_low))
    plt.show()
    plt.plot(np.fft.ifft(x_2000_filtered_low))
    plt.show()
    plt.plot(np.fft.ifft(x_200_filtered_High))
    plt.show()
    plt.xlim(0,500)
    plt.plot(np.fft.ifft(x_2000_filtered_High))
    plt.show()







def zplane(b, a, filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1];
    plt.xticks(ticks);
    plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k

#[b, a, w, h_dft] = filter_IIR_ellip(filter_order=2, fc_low= 900.0, fc_high = 1100.0, pass_band_ripple_db=1.0, stop_band_attn_db=40.0, fe = 20000)
#zplane(b, a)
#impulse_respond(b, a)
#FIR_overlap_save()
create_sinus()


