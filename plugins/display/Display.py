import time
import threading
import webbrowser
import inspect
import json

import numpy as np
import cherrypy
import os
import os.path

from pax import plugin

""" This plugin is an event display implemented as a web site loaded on a local python server. The web stuff is implemented in cherrypy. The idea is that this plugin opens a user's browser and directs it to the site index. From there all navigation and control is done within the browser. Web resources, including html files and javascript classes, are included in a subdirectory of this plugin's directory and must be present for the plugin to function properly.
"""

class DisplayPage(object):
	""" Cherrypy website for displaying event data. """

	def __init__(self, config):
		self.event = None
		self.top_array_map = config['topArrayMap']
		self.rdy = False
		self.my_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

	def SetEvent(self, event):
		self.rdy = False
		self.event = event

	@cherrypy.expose
	def getnext(self):
		self.rdy = True
		# wait until next event received, then reload waveform page
		while self.rdy == True:
			time.sleep(100 / 1000000.0)
		raise cherrypy.HTTPRedirect("/")

	@cherrypy.expose
	def index(self):
		if self.event == None:
			return """<html><head></head><body><h4>No event</h4></body></html>"""
		return open(os.path.join(self.my_directory, 'public', 'html', 'index.html'))

	@cherrypy.expose
	def pmtpattern(self):
		return open(os.path.join(self.my_directory, 'public', 'html', 'pmtpattern.html'))

	@cherrypy.expose
	def get_pmtpattern(self):
                geo = []
                total = np.sum(self.event.summed_waveform())
                for i in range(1, len(self.top_array_map)):
                        size = 0
                        row = 0
                        if self.event.pmt_waveform(i) != None:
                                row = np.sum(self.event.pmt_waveform(i))
                        size = (int)(row / (total / 248) * 10)
                        if size > 10:
                                size = 10
                        color = "red"
                        if size == 0:
                                color = "blue"
                                size = 1
                        geo.append({"x_axis": self.top_array_map[i]['x'],
                                    "y_axis": self.top_array_map[i]['y'],
                                    "radius": size,
                                    "color": color})
                ret = {'geometry': geo}
                return json.dumps(ret)

	@cherrypy.expose
	def get_waveform(self):
                points = []
                peaks = []
                for i in range(0, len(self.event.summed_waveform())):
                        points.append([i, self.event.summed_waveform()[i]])
		# now get peaks
                for i in range(0,len(self.event.S2s())):
                        peak = self.event.S2s()[i]
                        x = peak._get_var('top_and_bottom','position_of_max_in_waveform')
                        peaks.append(['s2',i,str(x),'%2f' % peak.area(), '%i' % peak.bounds()[0], '%i' % peak.bounds()[1]])
                for i in range(0,len(self.event.S1s())):
                        peak = self.event.S1s()[i]
                        x = self.event.S1s()[i]._get_var('top_and_bottom','position_of_max_in_waveform')
                        peaks.append(['s1',i,str(x),'%2f' % peak.area(), '%i' % peak.bounds()[0], '%i' % peak.bounds()[1]])

                ret = {'waveform': points, 'peaks': peaks}
                return json.dumps(ret)


class CherryThread(threading.Thread):
	"""Wrap the cherry webpage in its own thread. This way the server can run independently of the rest of pax"""

	def __init__(self, page):
		threading.Thread.__init__(self)
		self.page = page
		self.end = False

	def run(self):
		conf = {
			'/': {
				'tools.sessions.on': True,
				'tools.staticdir.root': os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
			},
			'/static': {
				'tools.staticdir.on': True,
				'tools.staticdir.dir': './public'
			},
		}
		cherrypy.quickstart(self.page, '/', conf)
		while self.end == False:
			time.sleep(100 / 1000000.0)
		self.exit()

	def stop(self):
		self.end = True


class DisplayServer(plugin.OutputPlugin):
	"""Plugin to start a display server on a local webhost and autolaunch a browser tab to view it"""

	def startup(self):
		self.page = DisplayPage(self.config)
		self.thread = CherryThread(self.page)
		self.thread.start()
		self.opened = False

	def write_event(self, event):
		if self.opened == False:
			url = "http://127.0.0.1:8080/"
			webbrowser.open(url, new=1, autoraise=True)
			self.opened = True
		self.page.SetEvent(event)
		while self.page.rdy == False:
			time.sleep(100 / 1000000.0)

	def shutdown(self):
		self.thread.stop()
