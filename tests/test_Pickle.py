__author__ = 'tunnell'

import unittest

from pax import pax
from pluginbase import PluginBase


class MyTestCase(unittest.TestCase):
    def setUp(self):
        conf = pax.get_configuration()
        plugin_source = pax.get_plugin_source(conf)
        self.obj = pax.instantiate('Pickle.WriteToPickleFile',
                                   plugin_source,
                                   conf)

    def test_something(self):
        self.obj.write_event({})


if __name__ == '__main__':
    unittest.main()
