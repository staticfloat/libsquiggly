from numpy import *
from scipy import *
from scipy.signal import *



def lsharm_window(window, freqs, weights=[1, .5, .5], fs=2.0):
	"""
	Perform least-harmonic squares analysis [1] on a window of data returning the
	dominant frequency component found in that window, as well as the error

	Parameters
	----------
	window : 1-D signal array
		Preferrably a numpy array
	freqs : list
		List of frequencies to search over (in Hz)
	weights : list
		List of weights for harmonics (default [1, .5, .5])
	fs : float
		Sampling rate in Hz (default 2.0)

	References
	----------
	[1] Qin Li; Atlas, L., "Time-variant least squares harmonic modeling,"
	Acoustics, Speech, and Signal Processing, 2003. Proceedings. (ICASSP '03)
	"""
	max_order = len(weights)
	N = len(window)

	P = zeros((len(freqs), 1))
	E = zeros((len(freqs), 1))
	NK = array([arange(N)*(order+1) for order in range(max_order)]).T
	for idx in range(len(freqs)):
		f0 = freqs[idx]

		nyq = min( max_order, int(floor((fs/2)/f0)))

		C = zeros( (nyq, 1), dtype=complex )
		for k in range(nyq):
			b = array([1, -exp(-2j*pi*(k+1)*f0/fs)])
			a = array([1, -2*cos(2*pi*(k+1)*f0/fs), 1])

			temp = lfilter( b, a, window)
			C[k] = exp( -2j*pi*(k+1)*f0/fs*(N-1)) * temp[-1] / sqrt(N)

		P[idx] = norm(dot(weights,C))

		# Calculate error
		T = exp( 1j*2*pi*f0/fs*NK ) / sqrt( N );
		sym = 2*real( dot(T,weights*abs(C)*exp(1j*angle(C)))).T
		E[idx] = norm(window - sym)**2


	max_idx = argmax(P)
	f0 = freqs[max_idx]
	E0 = E[max_idx]

	return f0, E0


