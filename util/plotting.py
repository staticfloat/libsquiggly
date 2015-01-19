from numpy import *
from scipy import *
from pylab import *

def imagesc(x, xstart=0, xlim=1, ystart=0, ylim=1, cbar=True):
	"""
	Plot a 2-D array normalized with no interpolation, on the given axis ranges.

	Parameters
	----------
	x : 2-D signal array
		Preferrably a numpy array
	{x,y}start : float
		The starting point for the {x,y} axis
	{x,y}lim : float
		The ending point for the {x,y} axis
	cbar : boolean
		Whether or not to show a colorbar (default True)
	"""
    xf = x.flatten()
    mini = abs(min(xf[isfinite(xf)]))
    maxi = abs(max(xf[isfinite(xf)]))
    clf()
    imshow((x - mini)/(maxi - mini), extent=(xstart, xstart + xlim, ystart, ystart + ylim), origin='lower', interpolation='nearest', cmap=cm.bone)
    axis('tight')
    if colorbar:
    	colorbar()
    draw()

def spectrogram(x, NFFT=256, fs=2.0, noverlap=128, windowfunc=hann, zeropadding="sandwich", cbar=True):
	"""
	Plot the spectrogram of a function using stft() and imagesc()

	See stft() and imagesc() for their respective parameter meanings
	"""
	P = STFT(x, NFFT, fs, noverlap, windowfunc, zeropadding)
	imagesc(10*log10(P), xlim=len(x)/fs, ylim=fs/2, cbar=cbar)
	return P