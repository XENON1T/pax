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


Installation prerequisites
==========================

We will assume you are installing on Mac or Linux here. For installation on Windows, 
see `this FAQ entry <http://xenon1t.github.io/pax/faq.html#can-i-set-up-pax-on-my-windows-machine>`_. 

Python 3
^^^^^^^^
Pax is written in Python 3; we recommend the
scientific python distribution `Anaconda <https://store.continuum.io/cshop/anaconda/>`_. To set this up in your linux home directory, do::

  wget http://repo.continuum.io/archive/Anaconda3-2.1.0-Linux-x86_64.sh
  bash Anaconda3-2.1.0-Linux-x86_64.sh
  export PATH=~/anaconda3/bin:$PATH  # If installed in default location

You need to point anaconda to the physics-specific packages of e.g. ROOT.  You can do this by putting the following in `~/.condarc`::

  channels:
    - http://conda.binstar.org/NLeSC
    - defaults

Alternatively, you can install Python 3.4 yourself (highly not recommended since ROOT probably won't work).  Note that you need write permission to the python distribution's directory and that Python 2 does not currently work.  

Additional python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^
Pax depends on several other python packages. While most python packages will install automatically,
some contain C++ code which must be compiled. If you have Anaconda you can get appropriate binaries 
for your platform using the `conda` tool::

  conda update conda
  conda create -n pax numpy scipy matplotlib pandas cython h5py numba pip snappy python-snappy
  source activate pax

Git and Github
^^^^^^^^^^^^^^

You must have `git` installed, which you can test by running at the command line::

  git

To get the code of pax, you need access to the private XENON1T github repository.  You can get this by emailing the `pax` developers. 


Installing pax
==============
There are two ways to install pax. Either method will automatically try to install any python packages pax depends on -- although we hope you installed the most important ones already (see above). If a module does not install, try using `conda` or `pip` to install missing dependencies. 

Option 1: user installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this option the pax code will be hidden away somewhere deep in python's directory structure so you won't accidentally look at it and learn our dangerous secrets. Also, you will only be able to update pax after we make a new release (about once a month). If this appeals to you, run::

    pip install git+https://github.com/XENON1T/pax.git
    
To update to a newer version, add ` --upgrade`` to the command above.


Option 2: developer installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this option you'll know where the code is, so you can look at it, play with it, and if you change anything you don't need to reinstall for your changes to take effect... However, be aware you are using the very latest ('nightly') version of pax, which may contain more bugs (but often contains less bugs). 

First `cd` to the folder you want pax to be installed. Then run::

    git clone https://github.com/XENON1T/pax.git
    cd pax
    python setup.py develop

To update to the latest pax, go to the directory with pax and run `git pull`. 

If you think you've made a useful change, you can contribute it! But please check the
`relevant documentation section`_ first.

.. _relevant documentation section: CONTRIBUTING.rst

Pax Tutorial
============
This section assumes that pax is installed, either from the instructions above
or via `the FAQ on running the code at LNGS <http://xenon1t.github.io/pax/faq.html#how-do-i-run-pax-at-lngs-on-xecluster>`_.

Now you should be able to run the command::

  paxer --help
    
from anywhere, which will give you a list of other command line options. If you have a graphical display, try `paxer --plot` and `paxer --plot_interactive`. You can select some data with the `--input` option::

  paxer --input /archive/data/xenon100/run_14/xe100_150213_1411/xe100_150213_1411_000000.xed --event 0 --plot

If you want to do something nonstandard, you can create your own configuration file
like `my_file.ini`::

   [pax]
   parent_configuration = 'XENON100'
   input = 'XED.ReadXED'
   output = [ 'Plotting.PlotChannelWaveforms3D',
              #'Plotting.PlotEventSummary',
            ]

   [Plotting]
   log_scale_entire_event = False
   #output_name = 'plots'  # Uncomment to write plot to disk


You can load this file with `paxer` by using the `config_path` option::

  paxer --config_path my_file.ini --input /archive/data/xenon100/run_14/xe100_150213_1411/xe100_150213_1411_000000.xed --event 0

You can uncomment the `output_dir` line to write the plots to a file.  Also, try
playing with what is in the list of outputs.  For example, you can reactivate
the `PlotEventSummary` that was produced in the first command from above.

There are many, many configuration options you can change. 
You can look through other configuration files such as `_base.ini` and `XENON100.ini` to get an idea of what you can do. Also, you can try to explore what plugins are included in pax. You can ask us questions on gitter (click button above) or email. Oh, and did we mention the the documentation at http://xenon1t.github.io/pax/?

.. [1] *Sneaky snappy workaround*: follow the instructions for 'developer installation', but just before `python setup.py develop`, edit `requirements.txt` in the pax folder and put a comment (`#`) sign in front of the `python-snappy>=0.5` line. Save the file and run `python setup.py develop`. Now you can use pax even if you couldn't install snappy. Har-har. If you use anything that involves the MongoDB interface, pax will crash; don't say we didn't warn you.
