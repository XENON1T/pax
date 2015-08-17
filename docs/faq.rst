==========================
Frequently Asked Questions
==========================

----------------------------------------
How do I run `pax` at LNGS on xecluster?
----------------------------------------

You should be able to run `pax` running at the shell the following command::

  export PATH=/home/tunnell/anaconda3/bin:$PATH

This can be added to your `.bashrc` to be run automatically when you login.  You
can check that it worked the following command::

  python -c "import pax; print(pax.__version__)"

Which should result in Python3 being used to print the pax version.


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
2. Install python-snappy and h5py from `Christoph Gohlke's page <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.
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
4. Go to bin and run ``python paxer --plot``. If it shows a waveform plot, you are done.
   If it complains about a missing module, I guess you shouldn't have ignored those warnings in step 3,
   install the missing module using easy_install, or Gohlke's page mentioned above.

Unfortunately, Windows + Python3 + Root is not exactly a winning team; if you use windows you'll be stuck
with the other 433 output formats we have.

-------------------------
How do I get ROOT working
-------------------------

ROOT is a dependency and you're expected to install it yourself.  That said, we offer the follow two scripts that should install it for you.  Be sure that your GCC compiler version is more than 4.8 because ROOT requires that now.  Most problems people experience with `pax` are related to problems within the ROOT system.  We're working with them to make it better.  In the meantime, try these scripts::

  
  source travis/linux_setup.sh  # Only run for Ubuntu
  source travis/install-ROOT.sh
  
If it fails for your system, feel free to try to fix it and let us know so we can update the script.  You can always look within the shell script to see what we're doing if you run into an issue.

* Just follow the instructions on `the PyROOT page <https://github.com/XENON1T/pax/blob/master/docs/pyroot.rst>`_.

-------------
Snappy on OSX
-------------

After instally `snappy` through MacPorts, please run::

  CFLAGS=-I/opt/local/include LDFLAGS=-L/opt/local/lib pip install python-snappy


------------------------------------------
How do I use pax to reduce raw data files?
------------------------------------------

First, you need to know the event numbers of the events you want. Use whatever analysis tool you like for this.

If it is just a few events, you can use the `--events` switch like so::

  paxer --config reduce_raw_data --input your_dataset --output your_reduced_dataset --event 3 45 937 ...

This will produce your reduced raw data set in your_reduced_dataset. It will be in the ZippedBSON format, as that's the only format that supports non-continuous event numbers (at least, for now).

If you want more than a few events, make a newline-separated file of event numbers like so::

  3
  45
  937
  ...

and save this as e.g. your_event_number_file.txt. Then use::

  paxer --config reduce_raw_data --input your_dataset --output your_reduced_dataset --event_numbers_file your_event_number_file.txt

If the dataset you want to reduce is not in the default input format (currently XED), you also want to give pax a configuration which overrides the read plugin with the read plugin of that format. For example, to reduce a ZippedBSON dataset, use::

  paxer --config ZippedBSON reduce_raw_data --input your_dataset --output your_reduced_dataset --event_numbers_file your_event_file.txt

------------------------------------------
How do I reduce the file size of my processed data?
------------------------------------------

For processed data, the default configuration is to store events, peaks and hits. Since there are usually many hits per event, they will take a lot of disk space. If you need to reduce the size size abd you do not need the hit information, check out the line found in `_base.ini` :

  fields_to_ignore = ['all_hits','sum_waveforms','channel_waveforms']

It may look like the hits are already ignored (`'all_hits'`) but there is another property called `'hits'` which is a peak property instead of an event property. Add both to `fields_to_ignore` and you're fine.
This is also the place to be if you want to reduce your file size in a different way. If you do not need some properties, just add them here and they will be ignored.


--------------------------------------------------------------
How do I use pax to generate files to be processed by Xerawdp?
--------------------------------------------------------------
Pax has an XED output plugin which you can use just like other output plugins. For example, to make an XED file containing simulated events, do `paxer --config XENON100 Simulation to_XED`.

The hard part is getting Xerawdp to read the XED file you produced:

* Make some folder on xecluster to contain everything.
* Make a subfolder `raw`, containing another subfolder `xe100_150726_1253` (I will keep using this dataset name, but you can of course put any date and time you want).
* In the `xe100_150726_1253` folder, put the XED file generated by pax. Rename it to `xe100_150726_1253.xed`.
* In the original folder, place the file `xed_test.xml` from pax's `examples`. Edit it to replace any occurrence of `/home/aalbers/xed_xdp_test` with the absolute path to your folder. 
* Ssh to `xecluster03`, then run `xerawdp -w xed_test.xml xe100_150726_1253`. The ROOT file will appear at `./processed/xe100_150726_1253/v0.4.5/xe100_150726_1253.root`.
* If you'd like to output the waveform of event 0 to .C instead, use `xerawdp -p -o xed_test.xml xe100_150726_1253 0`. The .C will appear in the current directory and can be opened by ROOT.

At the moment our hacked XML only works for one XED file (which can contain an arbitrary number of events though), and the instructions aren't very convenient. You're welcome to improve the situation!
