import tempfile
import os

datadir = '/data/xe100_110210_1926'

my_config = """
[pax]
parent_configuration = 'daq_injector'

[XED.XedInput]
filename = "%s"

[MongoDB.MongoDBFakeDAQOutput]
address = 'xedaqtest1:27017'
"""

files = [ os.path.join(base, f) 
          for base, _, files in os.walk(datadir) 
          for f in files if f.endswith(".xed") ] 

for i, filename in enumerate(files):
    outfd, outsock_path = tempfile.mkstemp()
    outsock = os.fdopen(outfd,'w')

    outsock.write(my_config % filename)
    print('paxer --config_path', outsock_path, '&')
    outsock.close()
    
    if i % 60 == 60 - 1:
        print('wait')
    
