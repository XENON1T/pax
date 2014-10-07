===============================
Processor for Analyzing XENON1T
===============================

The Processor for Analyzing XENON1T (PAX) is used for doing digital signal processing and other data processing on the XENON1T raw data.

* Free software: BSD license
* Documentation: http://xenon1t.github.io/pax/.

Quick Installation
------------------

Currently, we require Python 3.4.  Therefore, it is recommended to first install a Python virtual environment and specify where your Python binary is located. ::

    virtualenv -p python3.4 paxenv
    source paxenv/bin/activate

You can now install pax, which requires a github account ::

    git clone https://github.com/XENON1T/pax
    cd pax
    python setup.py install

Now you should be able to run the command 'paxer'.  For information on how to setup the code for contributing, please see the `relevant documentation section`_.

.. _relevant documentation section: CONTRIBUTING.rst

See paxer --help for more detailed usage information.

If you want to do something fancy, you can create your own configuration file like::

   [pax]
   parent_configuration = 'default'    # Inherit from the default configuration
   input = 'MongoDB.MongoDBInput'
   my_extra_transforms = ["PosSimple.PosRecWeightedSum"]
   output = ["Plotting.PlottingWaveform"]

and load it using paxer --config_path YOURFILE. We already have a few example configs available in config, which you can load using paxer --config NAME (with NAME, for example, XED_example or Mongo_example).

Features
--------

* Digital signal processing

 * Sum waveform for top, bottom, veto
 * Filtering with raised cosine filter
 * Peak finding of S1 and S2

* I/O

 * MongoDB (used online for DAQ)
 * XED (XENON100 format)
 * HDF5 (default output)
 * Pickle
 * Plots
 * DAQ injector

* Position reconstruction of events

 * Charge-weighted sum (x, y) reconstruction

* Interactive display

 * Interactive waveform with peaks annotated
 * PMT top layer hit pattern
 * Display is web browser-based. Allows navigation (next event, switch plot) within browser
