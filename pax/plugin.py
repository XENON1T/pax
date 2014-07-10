__author__ = 'tunnell'

import logging

class BasePlugin():
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

class InputPlugin():
    def __init__(self):
        self.i = 0

    def GetEventGenerator(self):
        try:
            while(1):
                self.i += 1
                self.log.debug("Loading event %d" % self.i)
                yield self.GetNextEvent()
        except StopIteration:
            self.log.debug("No more data")
            pass

    def GetNextEvent(self):
        """Get next event from the data source

        Raise a StopIteration when done
        """
        raise NotImplementedError()

class TransformPlugin():
    pass

class OutputPlugin():
    pass
