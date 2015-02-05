==========================
Frequently Asked Questions
==========================

----------------------------------------
How do I run `pax` at LNGS on xecluster?
----------------------------------------

You should be able to run `pax` running at the shell the following command::

  export PATH=/home/tunnell/anaconda3/bin:$PATH

This can be added to your `.bashrc` to be run automatically when you login.  You
can check that it worked by running `python` then::

  import pax

Which should result in version 3 of Python being used and you should not get an
error.

----------------------------------------
How do I run `pax` at Nikhef?
----------------------------------------

Running pax at Nikhef requires you to add python 3.4 to your path.
This can be done by running the following command from a shell::

  $ export PATH="/data/xenon/bartp/anaconda3/bin:$PATH"

This can be added to your `.bashrc` to be run automatically when you login.
After setting your environment you need a local copy of pax, clone it
from github by executing::

  $ git clone https://github.com/XENON1T/pax.git
  
After doing::

  $ cd pax/bin

Setup pax by executing::

  $ python setup.py develop
  
Pax should now work, test it by running::

  $ python paxer --plot

---------------------------------------
How do I run `git` at LNGS on xecluster
---------------------------------------

You can get `git` running by doing the following at the command line::

  export PATH=/home/kaminsky/software/bin:$PATH

The machines have old certificates, therefore you cannot run `git` in secure
mode.  Running the command is a workaround::

  export GIT_SSL_NO_VERIFY=true

These two export commands can be added to your `.bashrc` to be run automatically
when you login.


---------------------------------------
Can I set up pax on my windows machine?
---------------------------------------

Yes, in fact several of the developers do this, much to the sadness of the other developers...

1. Start with installing Anaconda for python 3 from their website. If the Anaconda website nags you for an email,
   enter your favourite fictional email address
2. Install python-snappy from `Christoph Gohlke's page <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.
   See installation instructions for the .whl file at the top of the page.
2. Get pax using Github's nice point-and-click Github For Windows application.
   Ignore the haters who insist on doing everything by command line.
3. Run ``python setup.py develop``.
   It may complain about some failed module installations.
   If it looks like an important module, try `conda install important_module`.
   If that fails, try `pip install important_module`
   If that fails, try `easy_install important_module`
   If that fails, look for important_module on `Christoph Gohlke's page <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_
   If that fails, reboot, sacrifice a goat, then try again.
4. Go to bin and run ``python paxer``. If it shows a waveform plot, you are done.
   If it complains about a missing module, I guess you shouldn't have ignored those warnings in step 3,
   install the missing module using easy_install, or Gohlke's page mentioned above.

Unfortunately, Windows + Python3 + Root is not exactly a winning team; if you use windows you'll be stuck
with the other 433 output formats we have.

---------------------------------------
How do I get Python 3.4 with ROOT working on Ubuntu 14?
---------------------------------------
* Just follow the instructions on `the PyROOT page <https://github.com/XENON1T/pax/blob/master/docs/pyroot.rst>`_.
