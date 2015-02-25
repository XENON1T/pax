===============================
Processor for Analyzing XENON
===============================

The Processor for Analyzing XENON (PAX) is used for doing digital signal
processing and other data processing on the XENON100/XENON1T raw data.

* Free software: BSD license
* Documentation: http://xenon1t.github.io/pax/.

.. image:: https://magnum.travis-ci.com/XENON1T/pax.svg?token=8i3psWNJAskpVjC6qe3w&branch=master
    :target: https://magnum.travis-ci.com/XENON1T/pax
.. image:: https://coveralls.io/repos/XENON1T/pax/badge.svg?branch=master
    :target: https://coveralls.io/r/XENON1T/pax?branch=master

Installation
=============

Installing requirements
-----------------------

We require Python 3.4 and several python modules. The easiest way to install these 
is to get a scientific python distribution such as `Anaconda <https://store.continuum.io/cshop/anaconda/>`_.
Use your package manager, or run::

  $ wget http://repo.continuum.io/anaconda3/Anaconda3-2.1.0-Linux-x86_64.sh
  $ bash Anaconda3-2.1.0-Linux-x86_64.sh
  $ export PATH=~/anaconda3/bin:$PATH  # If installed in default location


Alternatively, you can install Python 3.4 from the `python webpage <https://www.python.org/>`_ 
or your OS's package management system. You may have to install some additional modules manually.

You must separately install the `snappy compression library <https://code.google.com/p/snappy/>`_
and its python bindings. For Ubuntu you can do::

  $ sudo apt-get install libsnappy-dev

For installation on windows, see `the FAQ <http://xenon1t.github.io/pax/faq.html#can-i-set-up-pax-on-my-windows-machine>`_.


Installing pax
--------------

The code of pax is in a private github repository. You must first have `git`
installed, which you can test by running at the command line::

  git

Once you have access to the GitHub repository and `git` installed, run::

    pip install git+https://github.com/XENON1T/pax.git

This should automatically install any python modules pax depends on. 

Now you should be able to run the command 'paxer'.

For information on how to setup the code for contributing, please see the
`relevant documentation section`_.

.. _relevant documentation section: CONTRIBUTING.rst


First steps
===========

This section assumes that pax is installed, either from the instructions above
or via `the FAQ on running the code at LNGS <http://xenon1t.github.io/pax/faq.html#how-do-i-run-pax-at-lngs-on-xecluster>`_.

You can run the program by, for example, running::

  paxer --config XED --input /archive/data/xenon100/run_14/xe100_150213_1411/xe100_150213_1411_000000.xed --event 0 --plot

For other options and a list of command line arguments, please run::

  paxer --help

If you want to do something nonstandard, you can create your own configuration file
like `my_file.ini`::

   [pax]
    parent_configuration = 'XENON100'
    input = 'XED.XedInput'
    output = [ 'Plotting.PlotChannelWaveforms3D',
               #'Plotting.PlotEventSummary',
        ]

    [Plotting]
    log_scale_entire_event = False
    #output_dir = 'plots'  # Uncomment to write plot to disk



You can load this file with `paxer` by doing the following::

  paxer --config_path my_file.ini --input /archive/data/xenon100/run_14/xe100_150213_1411/xe100_150213_1411_000000.xed --event 0

You can uncomment the `output_dir` line to write the plots to a file.  Also, try
playing with what is in the list of outputs.  For example, you can reactivate
the `PlotEventSummary` that was produced in the first command from above.

At this point, you can look through other configuration files and explore what
plugins are in `pax` for doing more sophisticated things.

Features
========

Here is a list of some of the nice features you can play with:

* Digital signal processing

 * Sum waveform for top, bottom, veto
 * Filtering with raised cosine filter
 * Peak finding of S1 and S2

* I/O

 * ROOT
 * MongoDB (used online for DAQ)
 * Raw data from XENON100 and XENON1T (XED and Avro)
 * Plots

* Position reconstruction of events

 * Charge-weighted sum (x, y) reconstruction
 * (x, y) Reconstruction using chi-square-gamma minimization
 * Neural-net reconstruction


* Interactive display

 * Interactive waveform with peaks annotated
 * PMT top layer hit pattern
 * Display is web browser-based. Allows navigation (next event, switch plot)
   within browser
