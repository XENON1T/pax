__author__ = 'tunnell'

import unittest
from pluginbase import PluginBase

class MyTestCase(unittest.TestCase):
    def setUp(self):
        plugin_base = PluginBase(package='pax.plugins')
        searchpath = ['./plugins']
        plugin_source = plugin_base.make_plugin_source(searchpath=searchpath)


def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
