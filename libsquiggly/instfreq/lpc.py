from numpy import *
from scipy import *
from talkbox import lpc


def lpc_freqtrack(x, order=8, win_len=128, step=1, fs=2.0):
	"""
	Analyze a signal using linear predictive coding to discover the linear
	relationships between samples in windows of the signal. Use this linear model
	to determine dominant frequency component, and return the center frequency of
	that dominant component as the instantaneous frequency of the signal

	Parameters
	----------
	x : 1-D signal array
		Preferrably a numpy array
	order : int
		Order of linear model (default 8)
	win_len : int
		Length of window over which to build a single model (default 128)
	step : int
		Number of samples to step between windows (default 1)
	fs : float
		Sampling rate in Hz (default 2.0)

	Returns
	-------
	freq, err : 1-D signal array, 1-D signal array
		`freq` holds the instantaneous frequency estimates, according to `fs`
		`err` holds the instantaneous residual error
	"""

	nlen = int(ceil((len(x) - 1.0*win_len)/step))

	lpc_estimates = zeros(len(x)/step)
	lpc_error = zeros(len(x)/step)

	# Pad x with zeros
	pad_len = win_len/step
	x = hstack((zeros(pad_len/2), x, zeros(pad_len/2)))

	for i in range((len(x) - pad_len)/step):
		window = x[i*step:i*step + win_len]

		A, lpc_e, k = lpc(window, order)
		R = roots(A)
		lpc_f = angle(R[argmax(abs(R))])*fs/(2*pi)

		lpc_estimates[i] = lpc_f
		lpc_error[i] = 1.0/lpc_e

	return lpc_estimates, lpc_error
