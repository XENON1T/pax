"""Read/write event class from/to gzip-compressed pickle files.
"""

from pax import plugin

import gzip, re, glob
try:
    import cPickle as pickle
except:
    import pickle



class WriteToPickleFile(plugin.OutputPlugin):

    def write_event(self, event):
        with gzip.open(self.config['output_dir'] + '/' + str(event.event_number), 'wb') as file:
            pickle.dump(event, file)



class DirWithPickleFiles(plugin.InputPlugin):

    def get_events(self):
        files = glob.glob(self.config['input_dir'] + "/*")
        if len(files)==0:
            self.log.fatal("No files found in input directory %s!" % self.config['input_dir'])
        for file in files:
            if re.search('(\d+)$',file) is None:
                self.log.debug("Invalid file %s" % file)
                continue
            with gzip.open(file,'rb') as f:
                yield pickle.load(f)