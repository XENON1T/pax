__author__ = 'axel'
import h5py
import tables
import pandas as pd
filename    = '/home/axel/PycharmProjects/pax/tpc_kr_150410_8k.hdf5'
df = pd.read_hdf(filename,"Hit")