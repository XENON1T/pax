__author__ = 'tunnell'



import matplotlib.pyplot as plt

from pax import plugin
import objgraph

__author__ = 'tunnell'




class CreateGraphOfEventStructure(plugin.OutputPlugin):

    def write_event(self, event):
        objgraph.show_refs(event,
                           filename='sample-graph.png')