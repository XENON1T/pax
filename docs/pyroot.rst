===============================
PyROOT installation
===============================

The Processor for Analysing XENON (PAX) uses ROOT (http://root.cern.ch/drupal/) for the output of files. 
PyROOT is a Python extension module that allows the user to interact with any ROOT class from the Python interpreter.
For this purpose it is necessary to have the latest Python (3.4) version working with ROOT. This installation document 
shows the steps needed to do this correctly in an virtual environment on Ubuntu 14.

Installation
=============

We assume installation on Ubuntu 14 desktop with python3.4 pre-installed.

* To install virualenv with py3.4 run::
  
   sudo apt-get install python-pip
   sudo pip install virtualenv
   virtualenv -p /usr/bin/python3.4 $HOME/env/py34
   source ~/env/py34/bin/activate

Check that python3.4 starts when typing 'python' after activating the virtualenv. The virtualenv can be exited by typing 'deactivate'.

To get ROOT ready for compilation:

* Download the latest root: http://root.cern.ch/drupal/content/downloading-root
* Unpack and install dependencies with the command from: http://root.cern.ch/drupal/content/build-prerequisites::

   mkdir ~/env/py34/src
   cd ~/env/py34/src
   gzip -dc root_<version>.source.tar.gz | tar -xf -
   sudo apt-get install git dpkg-dev make g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev

* Search for the files Python.h and libpython3.4.so::

   sudo updatedb
   locate Python.h
   locate libpython3.4.so
   
* If they are not found run::

   sudo apt-get install python3.4-dev

* Now go to the root directory and set configure to the locations of the found files of the correct python version.::

   ./configure --enable-python --with-python-incdir=/usr/include/python3.4m --with-python-libdir=/usr/lib/python3.4/config-3.4m-x86_64-linux-gnu

where the Python include directory should point to the directory containing the Python.h file, and the library directory should contain libpython.x.y.[so][.dll], 
where 'x' and 'y' are the major and minor version number of Python,  respectively. Note that at the end of the configuration, the list of 
enabled features is printed: make sure that 'python' is listed among them. If you do not specify the inc and lib directories configure will try to use the environment 
variable $PYTHONDIR or alternatively will look in the standard locations.

* The output should now include::

    Checking for Python.h ... /usr/include/python3.4m
    Checking for python3.4, libpython3.4, libpython, python, or Python ... /usr/lib/

* Compile ROOT::

   make

TIP: When running the command make above, one can multi thread the build by doing make -jN for an N-core machine. For example, in a four core laptop, one could do make -j4.
The ROOT build should be successful

* Check that the PyROOT module was build correctly by looking for ROOT.py and libPyROOT.so::

   sudo updatedb
   locate ROOT.py
   locate libPyROOT.so

* Set this root as source for the env::

   echo source bin/thisroot.sh >> ~/env/py34/bin/activate

* Convert the code in ROOT.py from python 2 to python 3 with 2to3::

   2to3 -w ~/env/py34/src/root/lib/ROOT.py 

Start python and check that "import ROOT" works. If so happy programming.

Now every time you want to use PyROOT 3.4 activate the virtualenv::
   source ~/env/py34/bin/activate
