#!/usr/bin/env python

# Add '../' to the loading path so we can get at `libsquiggly`:
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from numpy import *
from scipy import *
from matplotlib.pyplot import *
from utils import *
from libsquiggly.util import *
from libsquiggly.tfr import *

# Generic test harness for our instantaneous-frequency estimation code
def test_tfr(x, f, sig_name):
	# Setup spectrogram parameters.
	NFFT = 512
	N = len(x)
	fs = 2.0
	t = linspace(0,N/fs,N)

	# Plot the spectrogram
	figure()
	P_stft = spectrogram(x, NFFT=NFFT, noverlap=NFFT-1, fs=fs, cbar=False)
	#plot(t, f)
	title("Spectrogram of " + sig_name)
	xlabel("Samples")
	ylabel("Frequency (normalized)")

	# Plot the GCKD
	figure()
	P_gckd = gckdgram(x, NFFT=NFFT, fs=fs, cbar=False)
	#plot(t, f)
	title("GCKD-gram of " + sig_name)
	xlabel("Samples")
	ylabel("Frequency (normalized)")




# 8 Ki-samples seems like a good number here
N = 8196

# Generate an FM sinusoid track
x_fm, f_fm = gen_fm_track(N, f0=.5, df=.35)
test_tfr(x_fm, f_fm, "frequency-modulated sinusoid")


# Generate a frequency-hopping sinusoid
x_hopping, f_hopping = gen_hopping_track(N, 7, fmin=.001, fmax=.8)
test_tfr(x_hopping, f_hopping, "frequency-hopping sinusoid")


# Generate a more wideband signal
x_wideband, f_wideband = gen_wideband_track(N, width=.1, f0=.5, df=.15)
test_tfr(x_wideband, f_wideband, "wideband nonstationary signal")

pause()
