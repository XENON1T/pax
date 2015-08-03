import tempfile
import os

datadir = '/data/'

my_config = """
[pax]
parent_configuration = 'XED'

[XED.ReadXED]
filename = "%s"

[MongoDB.MongoDBFakeDAQOutput]
address = 'xedaqtest1:27017'

[HDF5.HDF5Output]
hdf5file = '%s.h5'

[PosSimple.PosRecWeightedSum]

channels_to_use_for_reconstruction = 'top'
"""

files = [ os.path.join(base, f) 
          for base, _, files in os.walk(datadir) 
          for f in files if f.endswith(".xed") ] 

for i, filename in enumerate(files):
    outfd, outsock_path = tempfile.mkstemp()
    outsock = os.fdopen(outfd,'w')

    root = filename.split('/')[-1][:-4]
    outsock.write(my_config % (filename, root))
    print('paxer --config_path', outsock_path, '& #', filename)
    outsock.close()
    
    if i % 60 == 60 - 1:
        print('wait')
    
