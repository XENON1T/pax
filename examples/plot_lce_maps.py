import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from pax.utils import InterpolatingMap
from pax.core import data_file_name
from pax import units

map_file = 's2_xy_lce_map_XENON100_Xerawdp0.4.5'

try:
    os.mkdir(map_file)
except FileExistsError:
    pass

print("Reading map...")
maps = InterpolatingMap(data_file_name(map_file+'.json.gz'))
   
print("Plotting individual LCE maps")   
for m in tqdm(maps.map_names):
    # Reference plot of the XENON100 tpc radius
    r = 0.5 * 30.6 * units.cm
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(r*np.cos(theta), r*np.sin(theta), c='white')

    # Plot the LCE map
    maps.plot(map_name=m, to_file=os.path.join(map_file, m + '.png'))



# This is just to test the interpolation routines are working: you can skip it
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# print("Calculating & plotting overall LCE map")
# r = 15.3
# d = 0.2
# y, x = np.mgrid[slice(-r, r + d, d),
                # slice(-r, r + d, d)]
# z = np.zeros(x.shape)

# theta = np.linspace(0, 2*np.pi, 200)
# plt.plot(r*np.cos(theta), r*np.sin(theta), c='white')

# # Should vectorize this...     
# for i in tqdm(range(len(x))):
    # for j in range(len(y)):
        # for m in maps.map_names:
            # if m == 'total_LCE':
                # continue
            # z[i,j] += maps.get_value(x[i,j], y[i,j], map_name=m)
    
# # see http://matplotlib.org/examples/pylab_examples/pcolor_demo.html
# z[:-1:-1] 
# z_min, z_max = 0, np.abs(z).max()
# plt.pcolor(x, y, z, vmin=z_min, vmax=z_max)
# plt.title('pcolor')
# plt.axis([x.min(), x.max(), y.min(), y.max()])
# plt.colorbar()
# plt.xlabel('x (cm)')
# plt.ylabel('y (cm)')
# plt.title('S2Top LCE (from MC)')
# plt.savefig('summed_lce.png')