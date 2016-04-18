.. image:: docs/pax_logo_bw_cropped.png

The Processor for Analyzing XENON (PAX) is used for doing digital signal
processing and other data processing on the XENON100/XENON1T raw data.

* Free software: BSD license
* Documentation: http://xenon1t.github.io/pax/.

.. image:: https://travis-ci.org/XENON1T/pax.svg?branch=master
    :target: https://travis-ci.org/XENON1T/pax
.. image:: https://coveralls.io/repos/XENON1T/pax/badge.svg?branch=master
    :target: https://coveralls.io/r/XENON1T/pax?branch=master
.. image:: http://img.shields.io/badge/gitter-XENON1T/pax-blue.svg 
    :target: https://gitter.im/XENON1T/pax


Installation prerequisites
==========================

We will assume you are installing on Mac or Linux here. For installation on Windows, 
see `this FAQ entry <http://xenon1t.github.io/pax/faq.html#can-i-set-up-pax-on-my-windows-machine>`_. 

Instead of installing pax you can use pax from the central installation on xecluster or other cluster. See the `FAQ <https://github.com/XENON1T/pax/blob/master/docs/faq.rst>`_ for more details.

If you wish to develop pax, you can either follow these instructions from the beginning which installs your own copy of the Anaconda libraries, or use an existing Anaconda installation (skip to the "Setting Up the Anaconda Libraries" section below for this).


Installing Python 3 and Anaconda Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pax is written in Python; we recommend the scientific python distribution `Anaconda <https://store.continuum.io/cshop/anaconda/>`_. To install this in Linux do::

  wget http://repo.continuum.io/archive/Anaconda3-2.4.0-Linux-x86_64.sh  # Linux
  bash Anaconda3-2.4.0-Linux-x86_64.sh  # Say 'yes' to appending to .bashrc and specify the installation directory
  
--------------------------------

For Mac OS X, you should instead download the .sh file located here:

    http://repo.continuum.io/archive/Anaconda3-2.4.0-MacOSX-x86_64.sh
    
In the directory containing the file above, run the following command::

    bash Anaconda3-2.4.0-MacOSX-x86_64.sh  # Say 'yes' to appending to .bashrc and specify the installation directory
  
Setting Up the Anaconda Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For both Linux and Mac OS X::

  export PATH=~/anaconda3/bin:$PATH  # If installed in default location, otherwise replace with the 
                                     # path you specified above or the path of the central installation 

You need to point Anaconda to the physics-specific packages e.g. ROOT.  You can do this by running the following::

  conda config --add channels http://conda.anaconda.org/NLeSC  


Check that no other ROOT is seen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several people have encountered ROOT incompatibility issues when installing pax. Before continuing, we advise you to check that no other ROOT version that can be seen by your terminal, e.g try::

  root
  
Which should return ``bash: root: command not found``.  Also check that Python cannot see ROOT::

  python -c "import ROOT"

should say ``ImportError: No module named 'ROOT'``.  

If there are multiple versions of ROOT around, strange segfaults and other weird issues can result. This is an issue with ROOT and not pax.


Additional python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pax depends on several other python packages. While most python packages will install automatically,
some contain C++ code which must be compiled. With Anaconda you can get appropriate binaries 
for your platform using the `conda` tool as follows. Make sure to replace <environment_name> with the desired name of your environment (usually it is 'pax' for central installations) in the following command::

  conda update conda
  conda create -n <environment_name> python=3.4 root rootpy numpy scipy matplotlib pandas cython h5py numba pip python-snappy pytables scikit-learn psutil pymongo

If you do not want ROOT support, or have ROOT-related issues, you can leave out root and rootpy in the above command. Everything in pax, except of course ROOT I/O, will continue to work. You can also try root=6 for the newer version of ROOT.

If you want to use Ipython notebook within this environment (e.g. to import pax libraries), you should also add 'ipython ipython-notebook' to the list of libraries above.

Whenever you want to use `pax`, you have to run the following command to set it up your environment (containing all the dependencies)::
  
  source activate <environment_name>
  
You can put this in your `.bashrc` if you want it to be setup when you login. For the rest of the installation and to run pax, be sure to be inside this environment. There should be (<environment_name>) at the beginning of your command line.

-----------------------------------

Of course we assume you have a functional basic compilation environment. On a fresh Ubuntu you may have to do e.g.

  sudo apt-get install build-essential
  
to setup the necessary compilers.

On a Mac, please run the following to make sure that ROOT works::

  xcode-select --install


Git and Github
^^^^^^^^^^^^^^

You must have `git` installed, which you can test by running at the command line::

  git

To get the code of pax, you need access to the private XENON1T github repository.  You can get this by emailing the `pax` developers. 


Installing pax
==============

There are two ways to install pax. Either method will automatically try to install any python packages pax depends on -- although we hope you installed the most important ones already (see above). If a module does not install, try using `conda` or `pip` to install missing dependencies. 

**WARNING: If you are working with a central installation of Anaconda, e.g. on Midway, Stockholm, or xecluster, there is a risk of overwriting the central installation (we are still working out some permissions issues)!** To avoid this, make sure you are either using your own installation of Anaconda or have created a new environment by replacing <environment_name> in the instructions above.

Option 1: User installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this option the pax code will be hidden away somewhere deep in python's directory structure so you won't accidentally look at it and learn our dangerous secrets. Also, you will only be able to update pax after we make a new release (about once a month). If this appeals to you, run::

    pip install git+https://github.com/XENON1T/pax.git
    
To update to a newer version, add ` --upgrade`` to the command above (or just run the same command again).


Option 2: Developer installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this option you'll know where the code is, so you can look at it, play with it, and if you change anything you don't need to reinstall for your changes to take effect... However, be aware you are using the very latest ('nightly') version of pax, which may contain more bugs (but often contains less bugs). 

First `cd` to the folder you want pax to be installed. Then run::

    git clone https://github.com/XENON1T/pax.git
    source activate <environment_name>  # Make sure you specify your own environment 
                                        # when using a central installation of Anaconda
    cd pax
    python setup.py develop

To update to the latest pax, go to the directory with pax and run `git pull`. 

If you think you've made a useful change, you can contribute it! But please check the
`relevant documentation section`_ first.

.. _relevant documentation section: CONTRIBUTING.rst

To check if your installation is working
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Execute::

    paxer --version

or

    paxer --input ~/pax/pax/data/xe100_120402_2000_000000.xed --stop_after 1 --plot --config="XENON100"

You should see a nice plot of a XENON100 event.


Pax Tutorial
============
This section assumes that pax is installed, either from the instructions above
or via `the FAQ on running the code at LNGS <http://xenon1t.github.io/pax/faq.html#how-do-i-run-pax-at-lngs-on-xecluster>`_.

Now you should be able to run the command::

  paxer --help
    
from anywhere, which will give you a list of other command line options. If you have a graphical display, try `paxer --plot` and `paxer --plot_interactive`. You can select some data with the `--input` option::

  paxer --input /archive/data/xenon100/run_14/xe100_150213_1411/xe100_150213_1411_000000.xed --event 0 --plot --config="XENON100"

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
