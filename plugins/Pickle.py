from pax import plugin

# try:
#    import cPickle as pickle
# except:
import pickle

__author__ = 'tunnell'


class WriteToPickleFile(plugin.OutputPlugin):

    def __init__(self, config):
        plugin.OutputPlugin.__init__(self, config)

        self.log.debug("Writing pickled data to %s" % config['picklefile'])
        self.myfile = open(config['picklefile'],
                         'wb')

    def __del__(self):
        self.log.debug("Closing %s" % self.config['picklefile'])
        self.myfile.close()

    def write_event(self, event):
        print(event)
        self.log.debug('Pickling event')
        pickle.dump(event, self.myfile)
