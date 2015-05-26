from numpy import *


def energy(x, already_demeaned=False):
    """
    Calculate the energy in x, demeaning if necessary

    Parameters
    ----------
    x : 1-D signal array
        The actual timeseries window to calculate the energy over
    already_demeaned : bool (default: False)
        If the signal has already been demeaned, don't bother calculating the mean again

    Return values
    -------------
    energy : float
        The calculated energy of the input signal
    """
    if not already_demeaned:
        z = x - mean(x)
        return sqrt(real(vdot(z, z)))
    else:
        return sqrt(real(vdot(x, x)))


def matched_filter( data, h, step = 1, mode="same" ):
    """
    Perform matched filtering between a datastream and an array representing the template filter

    Parameters
    ----------
    data : 1-D signal array stream (nditer)
        The actual timeseries to filter through.  This should be an iterator, so if you have
        a numpy array called x, pass in nditer(x) to this function
    h : 1-D signal array
        The filter to be used as the template filter to be searched for
    step : int (default: 1)
        To save computation time, the output can be pre-emptively decimated before being passed
        out of this function.  Set this number to an integer greater than 1 to skip output samples.
    mode : string
        Similar to numpy.convolve(); output length should be "same" or "valid" to disable/enable
        chomping of output that is due to transient response at the beginning/end of a stream.
        Note that since this function assumes an infinite stream, it only bothers with the beginning

    Return values
    -------------
    data_hat : 1-D signal array stream
        This function acts as a generator, calculating samples on demand.  Use an iteration
        to pull all the data out of it.
    """
    if not mode in ["same", "valid"]:
        raise ValueError("Invalid mode, must be either 'same' or 'valid'")

    # Demean/normalize h first, so that we never have to do it again
    h = copy(h) - mean(h)
    h = h/energy(h, already_demeaned = True)
    h_len = len(h)

    # Grab the first element
    x = next(data)

    # Load in a single step
    buff = zeros((len(h),), x.dtype)
    buff[-1] = x
    for idx in range(step-1):
        buff = roll(buff, -1, 0)
        buff[-1] = next(data)

    # if mode is "valid", skip over any extra samples that we need to to skip the transient response
    if mode == "valid":
        for idx in range(len(h) - (step-1)):
            buff = roll(buff, -1, 0)
            buff[-1] = next(data)

    # Calculate dot product at each point, normalizing out x's energy
    idx = 0
    while True:
        x_slice = buff - mean(buff)
        x_slice_energy = energy(x_slice)

        if x_slice_energy == 0.0:
            yield 0.0
        else:
            output = vdot(x_slice, h)/x_slice_energy
            yield output
        
        for idx in range(step):
            buff = roll(buff, -1, 0)
            buff[-1] = next(data)