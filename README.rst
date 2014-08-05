===============================
Processor for Analyzing XENON1T
===============================

PAX is used for doing digital signal processing and other data processing on the XENON1T raw data

* Free software: BSD license
* Documentation: http://xenon1t.github.io/pax/.

Quick Installation
------------------

It is recommended to first install a python virtual environment. ::
    
        virtualenv pax_env
        source pax_env/bin/activate
  
Pull from github (and replace ${USER} with your username)::

        git clone https://${USER}@github.com/XENON1T/pax pax
        # Input your password when prompted

Install pax ::

        cd pax
        python setup.py install

Run the code ::

        python bin/paxit.py
  
You can edit the bin file (it's recommended to copy it first) to put in your own custom parameters. A list of defaults is in pax/defaults.ini.

Features
--------

* Digital signal processing
 * Sum waveform for top, bottom, veto
 * Filtering with raised cosine filter
 * Peak finding of S1 and S2
* I/O
 * MongoDB (used online for DAQ)
 * XED (XENON100 format)
 * ROOT
 * Pickle
 * Plots
* Position reconstruction of events
 * Demo reconstruction algorithm of charge-weighted sum
* Interactive display
 * Interactive waveform with peaks annotated
 * PMT top layer hit pattern
 * Display is web browser-based. Allows navigation (next event, switch plot) within browser
