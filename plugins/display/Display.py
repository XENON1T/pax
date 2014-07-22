import webbrowser
import time
from pax import plugin, pax
import numpy as np
import threading
import cherrypy
import os, os.path 
import webbrowser
import inspect
import json
from operator import itemgetter

class DisplayPage(object):
    """ Cherrypy website for displaying event data. """
    def __init__(self, config):                
        self.event=None
        self.topArrayMap = config['topArrayMap']
        self.rdy=False

    def SetEvent(self,event):
        self.rdy=False
        self.event=event

    @cherrypy.expose
    def getnext(self):
        self.rdy=True
        #wait until next event received, then reload waveform page
        while self.rdy == True:
            time.sleep(100/1000000.0)
        raise cherrypy.HTTPRedirect("/")
        
    @cherrypy.expose
    def index(self):
        if self.event == None:
            return """<html><head></head><body><h4>No event</h4></body></html>"""  
        direc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        return open(os.path.join(direc,'public','html','index.html'))
    
    @cherrypy.expose
    def pmtpattern(self):
        return """<html>
        <head></head>
        <body>
           <a href="/">Home</a>
        </body>
        </html>"""
    
    @cherrypy.expose
    def get_waveform(self):
        points = []
        for i in range(0, len(self.event['processed_waveforms']['top_and_bottom'])):
            points.append([ i, self.event['processed_waveforms']['top_and_bottom'][i]])
        #now get peaks
        s1Rank = s2Rank = 0
        peaks = []
        for peak in sorted(self.event['peaks'], key=lambda x: x['top_and_bottom']['area']):
            if peak['rejected']:
                continue
            x = peak['top_and_bottom']['position_of_max_in_waveform']
            ptype = 's1'
            rank  = s1Rank
            if peak['peak_type'] == 'large_s2' or peak ['peak_type'] == 'small_s2':
                ptype = 's2'
                rank = s2Rank
                s2Rank+=1
                
            else:
                s1Rank+=1
            peaks.append( [ptype, rank, str(x), '%2f' % peak['top_and_bottom']['area'], '%2f' % peak['left'], '%2f' % peak['right']])
                
        ret = {'waveform': points, 'peaks': peaks}
        return json.dumps(ret)


class CherryThread(threading.Thread):
    """Wrap the cherry webpage in its own thread. This way the server can run independently of the rest of pax"""
    def __init__(self,page):
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
            time.sleep(100/1000000.0)
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
            time.sleep(100/1000000.0)

    def shutdown(self):
        self.thread.stop()


 
