.. image:: docs/pax_logo_bw_cropped.png

The Processor for Analyzing XENON (PAX) is used for doing digital signal
processing and other data processing on the XENON100/XENON1T raw data.

* Free software: BSD license
* Documentation: http://xenon1t.github.io/pax/.

.. image:: https://magnum.travis-ci.com/XENON1T/pax.svg?token=8i3psWNJAskpVjC6qe3w&branch=master
    :target: https://magnum.travis-ci.com/XENON1T/pax
.. image:: https://coveralls.io/repos/XENON1T/pax/badge.svg?branch=master
    :target: https://coveralls.io/r/XENON1T/pax?branch=master
.. image:: http://img.shields.io/badge/gitter-XENON1T/pax-blue.svg 
    :target: https://gitter.im/XENON1T/pax

Installation
=============

Installing requirements
-----------------------

We use Python 3.4 and several python modules. To allow people to run `pax` from
their home directory and to ease people using pax, we recommend they setup the
scientific python distribution `Anaconda <https://store.continuum.io/cshop/anaconda/>`_.
This should make it easier to install our code than typical experiments, and if this
isn't true, please let us know (See `relevant documentation section`_).  To set this
up, run::

  wget http://repo.continuum.io/anaconda3/Anaconda3-2.1.0-Linux-x86_64.sh
  bash Anaconda3-2.1.0-Linux-x86_64.sh
  export PATH=~/anaconda3/bin:$PATH  # If installed in default location
  conda update conda
  conda install numpy scipy matplotlib pandas pytables cython h5py numba pip scikit-learn # optional but handy

Alternatively, you can install Python 3.4 from the `python webpage <https://www.python.org/>`_ 
or your OS's package management system. See the FAQ for more information.

Though most of our dependencies are solved by using Anaconda, there is one
dependency that often cannot be installed on olders machines. You must separately 
install the `snappy compression library <https://code.google.com/p/snappy/>`_,
which is C++ code that must be compiled and is used for raw data access. If 
you're using Ubuntu and have super user permissions, you could just install the libsnappy-dev package.  
However, we recommend you do the following::

  wget https://snappy.googlecode.com/files/snappy-1.1.1.tar.gz
  tar xvfz snappy-1.1.1.tar.gz 
  cd snappy-1.1.1
  ./configure --prefix=`conda info --root`
  make install
  cd ~
  CFLAGS=-I`conda info --root`/include LDFLAGS=-L`conda info --root`/lib pip install python-snappy
  
You should now be able to run the following command::

  python -m snappy

For installation on Windows, see `the FAQ <http://xenon1t.github.io/pax/faq.html#can-i-set-up-pax-on-my-windows-machine>`_. 
Also within the FAQs, you can find other useful hints.

Installing pax
--------------

The code of pax is in a private github repository. You must first have `git`
installed, which you can test by running at the command line::

  git

You must have access to the private XENON1T github repository.  You can get this by emailing the `pax` developers.  Once you have access to the GitHub repository and `git` installed, run::

    pip install git+https://github.com/XENON1T/pax.git

This should automatically install any python modules pax depends on. 

Now you should be able to run the command 'paxer'.

If you want to modify the code (i.e., have the source code), please see the
`relevant documentation section`_.

.. _relevant documentation section: CONTRIBUTING.rst


First steps
===========

This section assumes that pax is installed, either from the instructions above
or via `the FAQ on running the code at LNGS <http://xenon1t.github.io/pax/faq.html#how-do-i-run-pax-at-lngs-on-xecluster>`_.

You can run the program by, for example, running::

  paxer --input /archive/data/xenon100/run_14/xe100_150213_1411/xe100_150213_1411_000000.xed --event 0 --plot

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
