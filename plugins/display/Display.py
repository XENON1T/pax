import webbrowser
import time
from pax import plugin, pax
import numpy as np
import threading
import cherrypy
import os, os.path 
import csv
import inspect

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
        while self.rdy == True:
            time.sleep(100/1000000.0)
        raise cherrypy.HTTPRedirect("/")

    @cherrypy.expose
    def index(self):
        if self.event == None:
            return """<html><head></head><body><h4>No event</h4></body></html>"""  
        
        direc = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        path = os.path.join(direc,'public','waveform.csv') 
        print(path)
        print("HERE!")
        
        tempfile = open(path,'w')
        for i in range(0, len(self.event['processed_waveforms']['top_and_bottom'])):
            line = str(i) + ',' + str(self.event['processed_waveforms']['top_and_bottom'][i]) + '\n'
            tempfile.write(line)
            
        tempfile.close()

        return """<html>
        <head>
        <script src="/static/js/dygraph-combined.js"></script>
        </head>
        <body>
             <h4>Pax event display</h4>
             <h4>This event has %(numpeaks)s peaks</h4>
             <div id="plot" style="width:98%%"></div>
             <a href="/pmtpattern">PMT Hit Pattern</a>
             <a href="/getnext">Next Event</a>
        </body>
        <script>
            new Dygraph(plot, "/static/waveform.csv",{
                 legend: 'always',
                 title: 'waveform',
                 showRoller: true,
                 rollPeriod: 10,
                 yLabel: "p.e.",
                 xLabel: "bin (10ns)",
            });
        </script>
        </html>""" %{ "numpeaks": len(self.event['peaks']),
                       }
    
    @cherrypy.expose
    def pmtpattern(self):
        return """<html>
        <head></head>
        <body>
           <a href="/">Home</a>
        </body>
        </html>"""


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
#        cherrypy.quickstart(self.page)

    def write_event(self, event):                
        self.page.SetEvent(event)
        while self.page.rdy == False:
            time.sleep(100/1000000.0)
#        self.page.ClearNextEvent

    def shutdown(self):
        self.thread.stop()


 
