"""Pickle events

Write event class to a file.
"""

from pax import plugin

try:
    import cPickle as pickle
except:
    import pickle


class WriteToPickleFile(plugin.OutputPlugin):

    def startup(self):
        self.log.debug("Writing pickled data to %s" %
                       self.config['picklefile'])
        self.file = open(self.config['picklefile'],
                         'wb')

    def write_event(self, event):
        self.log.debug('Pickling event')
        pickle.dump(event,
                    self.file)

    def shutdown(self):
        self.log.debug("Closing %s" % self.config['picklefile'])
        self.file.close()
