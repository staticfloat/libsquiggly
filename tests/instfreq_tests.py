#!/usr/bin/env python

# Add '../' to the loading path so we can get at `libsquiggly`:
import sys, os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../'))

from numpy import *
from scipy import *
from pylab import *
from testutils import *
from libsquiggly.util import *
from libsquiggly.instfreq import *
from libsquiggly.tfr import *


# Generic test harness for our instantaneous-frequency estimation code
def test_instfreq(x, f, sig_name):
	# Setup spectrogram parameters.  Note that we use spectrogram to calculate
	# the STFT for TFR-based estimation methods such as max_peak and peaktracing
	NFFT=256
	N = len(x)
	fs = 2.0
	t = linspace(0,N/fs,N)

	# Always plot into a new figure
	figure()
	P = spectrogram(x, NFFT=NFFT, noverlap=NFFT-1, fs=fs, cbar=False)

	# Calculate frequency tracks.  Note that TFR-based estimation methods such as
	# max_peak and anything with peaktracing in the name are indexed in time by
	# STFT time bins.  In this case, noverlap = NFFT-1, therefore the time bins
	# are just samples, but in general you won't have such densely overlapped bins
	# in time, and thus you will need to calculate a proper time base.
	f_maxpeak = max_peak(P, fs)
	f_guided = guided_peaktracing(P, guide=f_maxpeak, sigma=15.0, fs=fs)
	f_windowed = windowed_peaktracing(P, sigma=15.0, fs=fs)
	f_lpc, lpc_error = lpc_freqtrack(x, order=2, fs=fs)

	# Plot everything over the spectrogram
	plot(t, f)
	plot(t, f_maxpeak)
	plot(t, f_guided)
	plot(t, f_windowed)
	plot(t, f_lpc)

	# Make it look all "professional"
	title("Spectrogram of " + sig_name)
	xlabel("Samples")
	ylabel("Frequency (normalized)")
	legend(["Ground truth", "max_peak", "guided_peaktracing", "windowed_peaktracing", "lpc"])



# 8 Ki-samples seems like a good number here
N = 8196

# Generate an FM sinusoid track
x_fm, f_fm = gen_fm_track(N, f0=.5, df=.35)
test_instfreq(x_fm, f_fm, "frequency-modulated sinusoid")


# Generate a frequency-hopping sinusoid
x_hopping, f_hopping = gen_hopping_track(N, 7, fmin=.2, fmax=.8)
test_instfreq(x_hopping, f_hopping, "frequency-hopping sinusoid")


# Generate a more wideband signal
x_wideband, f_wideband = gen_wideband_track(N, width=.1, f0=.5, df=.15)
test_instfreq(x_wideband, f_wideband, "wideband nonstationary signal")


pause()
