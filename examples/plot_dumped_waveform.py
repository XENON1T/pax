import numpy as np
import matplotlib.pyplot as plt

with open('waveforms/0.dat','rb') as input:
    wave = np.fromfile(input)
    print("Wave is %s samples long, type is %s, data type %s" % (
        len(wave), type(wave), wave.dtype
    ))
    plt.plot(wave)
    plt.show()