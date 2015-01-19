from numpy import *
from scipy import *
from scipy.signal import *

def STFT(x, NFFT=256, fs=2.0, noverlap=128, windowfunc=hann, zeropadding="sandwich"):
	"""
	Calculate the short-time fourier transform of an input signal x

	Parameters
	----------
	x : 1-D signal array
		Preferrably a numpy array
	NFFT : int 
		Desired frequency resolution in bins (default 256)
	fs : float
		Sampling rate in Hz (default 2.0)
	noverlap : int
		Number of samples to overlap analysis windows (default 128)
	windowfunc : function
		Window function to apply before fft analysis (default `scipy.signal.hann`)
	zeropadding : string
		What kind of zero-padding to apply to `x` before analysis.  One of:
		* "after": `pad_amnt` zeros are added to the end of `x`
		* "sandwich": `NFFT/2` zeros are added to beginning of `x`, followed by
			`NFFT/2 + pad_amnt` zeros added to the end of `x`
		`pad_amnt` is the number of samples required to pad `x` to the nearest
		integer multiple of `NFFT`. Default padding strategy is "sandwich", which
		yields nice properties when lining up time indices
	"""

	if noverlap >= NFFT:
		raise ValueError("`noverlap` (%d) must be less than or equal to `NFFT` (%d)"%(noverlap, NFFT))
	step = (NFFT-noverlap)
	pad_amnt = step - len(x)%step

	nlen = len(x)/step
	P = empty((NFFT/2,nlen))
	window = windowfunc(NFFT)

	if zeropadding == "sandwich":
		x = hstack((zeros(NFFT/2), x, zeros(NFFT/2 + pad_amnt)))
	elif zeropadding == "after":
		x = hstack((x, zeros(pad_amnt)))
	else:
		raise ValueError("Unrecognized `zeropadding` value: " + zeropadding)

	for idx in xrange(nlen):
		#if nlen > 5000 and idx % 441 == 0:
		#	sys.stdout.write("\r[stft] %.2fs"%((idx*step + offset)/fs))
		#	sys.stdout.flush()
		x_slice = x[idx*step:idx*step + NFFT]
		P[:,idx] = abs(fft(x_slice*window)[:NFFT/2])

	#if nlen > 5000:
	#	sys.stdout.write("\r[stft] %.2fs\n"%((nlen*step + offset)/fs))
	#	sys.stdout.flush()
	return P

