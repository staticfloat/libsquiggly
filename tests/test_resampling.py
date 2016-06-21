#!/usr/bin/env python

# Add '../' to the loading path so we can get at `libsquiggly`:
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from numpy import *
from scipy import *
from scipy.signal import *
from matplotlib.pyplot import *
from libsquiggly.util import *
from libsquiggly.resampling import *
from libsquiggly.resampling.upfirdn import upfirdn


def test_resampling(fs_start, fs_end, N=42):
	t_start  = arange(fs_start)*1.0/fs_start
	t_end    = arange(fs_end)*1.0/fs_end

	# Generate a super-super-super oversampled sinusoid
	x_sin = sin(2*pi*10.0*t_start)
	y_sin = resample_cascade(x_sin, fs_start, fs_end, N=N)
	z_sin = resample(x_sin, fs_end)
	figure()
	plot(t_start, x_sin)
	plot(t_end, y_sin)
	plot(t_end, z_sin)
	xlabel("Time (s)")
	ylabel("Amplitude")
	title("Sinusoidal signal (%.2fHz -> %.2fHz)"%(fs_start, fs_end))
	legend(["Original", "Cascaded", "Classical"])

	# Now try square wave
	x_sqr = acollect(map(lambda t: int(t*8)%2, t_start))
	y_sqr = resample_cascade(x_sqr, fs_start, fs_end, N=N)
	z_sqr = resample(x_sqr, fs_end)
	figure()
	plot(t_start, x_sqr)
	plot(t_end, y_sqr)
	plot(t_end, z_sqr)
	xlabel("Time (s)")
	ylabel("Amplitude")
	title("Square-wave signal (%.2fHz -> %.2fHz)"%(fs_start, fs_end))
	legend(["Original", "Cascaded", "Classical"])

	# Now sawtooth wave!
	x_tri = sawtooth(8*2*pi*t_start)
	y_tri = resample_cascade(x_tri, fs_start, fs_end, N=N)
	z_tri = resample(x_tri, fs_end)
	figure()
	plot(t_start, x_tri)
	plot(t_end, y_tri)
	plot(t_end, z_tri)
	xlabel("Time (s)")
	ylabel("Amplitude")
	title("Sawtooth signal (%.2fHz -> %.2fHz)"%(fs_start, fs_end))
	legend(["Original", "Cascaded", "Classical"])


print("Running resampling tests...")

# Test powers of two, downsampling and upsampling
print("Running 8192 -> 128 -> 8192 tests...")
fs_start = 8192
fs_end   = 128
test_resampling(fs_start, fs_end)
test_resampling(fs_end, fs_start)


# Test non-powers of two
print("Running 8192 -> 100 -> 8192 tests...")
fs_start = 8192
fs_end   = 100
test_resampling(fs_start, fs_end)
test_resampling(fs_end, fs_start)


pause()
