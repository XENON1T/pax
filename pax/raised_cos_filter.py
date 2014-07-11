import numpy as np

def rcosfilter(N, alpha, Ts, Fs):
    """
Generates a raised cosine (RC) filter (FIR) impulse response.
Parameters
----------
N : int
Length of the filter in samples.

alpha: float
Roll off factor (Valid values are [0, 1]).

Ts : float
Symbol period in seconds.

Fs : float
Sampling Rate in Hz.

Returns
-------

h_rc : 1-D ndarray (float)
Impulse response of the raised cosine filter.

time_idx : 1-D ndarray (float)
Array containing the time indices, in seconds, for the impulse response.
"""

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rc = np.zeros(N, dtype=float)
        
    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rc[x] = 1.0
        elif alpha != 0 and t == Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        elif alpha != 0 and t == -Ts/(2*alpha):
            h_rc[x] = (np.pi/4)*(np.sin(np.pi*t/Ts)/(np.pi*t/Ts))
        else:
            h_rc[x] = (np.sin(np.pi*t/Ts)/(np.pi*t/Ts))* \
                    (np.cos(np.pi*alpha*t/Ts)/(1-(((2*alpha*t)/Ts)*((2*alpha*t)/Ts))))
    
    return time_idx, h_rc

def filter(y, N = 31, alpha = 0.2, Ts = 1, Fs = 10):
    """

1. Calls the function to generate a raised cosine impulse response (ie signal)
2. Performs a convolution to use the filter to smoothen the signal 

Parameters
----------
N : int
Length of the filter in samples.

alpha: float
Roll off factor (Valid values are [0, 1]).

Ts : float
Symbol period in seconds.

Fs : float
Sampling Rate in Hz.

y : float (list)
An array with the signal that will be filtered.

Returns
-------

filtered: float
An array containing the smoothened the signal.

"""
    rcos_t, rcos_i = rcosfilter(N, alpha, Ts, Fs)
    filtered = np.convolve(y, rcos_i/rcos_i.sum(), 'same') # see numpy.convolve manual page for other modes than 'same'
    
    return filtered



"""
Below is an example which generates a sinus signal, and adds some normal distributed noise to it.
Then the function filter will be called to smoothen the signal. A plot will be generated thereafter.
"""


npts = 1024
end = 8
sigma = 0.5
x = np.linspace(0,end,npts)
r = np.random.normal(scale = sigma, size=(npts))
s = np.sin(2*np.pi*x)#+np.sin(4*2*np.pi*x)
y = s + r

z = filter(y)

import matplotlib.pyplot as plt
plt.plot(y)
plt.plot(z)
plt.show()




