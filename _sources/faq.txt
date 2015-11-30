==========================
Installation and setup
==========================


Setup PyCharm
~~~~~~~~~~~~~

See http://docs.continuum.io/anaconda/ide_integration#pycharm

How do I set up `pax` at LNGS on xecluster?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can either use a pax someone else has set up, or set up your own. To use Chris' pax, use the following command:

  export PATH=/home/tunnell/anaconda3/bin:$PATH
  source activate pax

This can be added to your `.bashrc` to be run automatically when you login.

To set up a developer installation of pax in your xecluster home directory, first follow the steps for installing anaconda and the required packages in the main readme (I didn't try to install snappy but used the workaround). Then add the following to your .bashrc::

    # use git in totally insecure mode
    export PATH=/home/kaminsky/software/bin:$PATH
    export GIT_SSL_NO_VERIFY=true

Now follow these steps::

    cd ~
    git clone https://github.com/XENON1T/pax.git
    wget http://curl.haxx.se/ca/cacert.pem ~/cacert.pem
    mkdir .pip
    cp /home/aalbers/.pip/pip.conf ~/.pip
    pip install avro-python3 flake8 prettytable tqdm pymongo
    cd pax
    python setup.py develop

If it complains about any more missing modules, install it using pip install. 

Whichever way you want to use pax, you check that it worked using the following command::

  paxer --version

which should result in Python3 being used to print the pax version.



How do I run `pax` at Nikhef?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to above. If you want, you can use the pax Jelle set up by adding::

  $ export PATH="/data/xenon/anaconda/bin:/data/xenon/pax:$PATH"

This can be added to your `.bashrc` to be run automatically when you login.


Can I set up pax on my windows machine?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


How do I get pyROOT working
~~~~~~~~~~~~~~~~~~~~~~~~~~~
To use ROOT output from pax, you need to make sure pyROOT works. You have several options
  * **Use python 2.** In this case, pyROOT should work (for windows only on ROOT 5). Just make sure the `PYTHONPATH` environment variable (which controls where python searches for libraries) is set to the directory containing ROOT.py. 
  * **Use Daniela's binaries with conda** Daniela has packaged ROOT + pyROOT + rootpy in a conda package. Instructions on how to use it are here: `https://github.com/rootpy/rootpy/issues/642#issuecomment-137389446`
  * **Use the travis scripts** Before Daniela's package, we made scripts to install ROOT on travis. Be sure that your GCC compiler version is more than 4.8 because ROOT requires that now. The scripts are here::
  
  source travis/linux_setup.sh  # Only run for Ubuntu
  source travis/install-ROOT.sh

  * **Use Sander's instructions** Before those scripts, Sander made `these instructions <https://github.com/XENON1T/pax/blob/master/docs/pyroot.rst>`_.


How do I install Snappy on OSX?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After instally `snappy` through MacPorts, please run::

  CFLAGS=-I/opt/local/include LDFLAGS=-L/opt/local/lib pip install python-snappy

=======
Usage
=======


How do I analyze some specific XENON100 events with pax?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the Dump_XENON100_events tool available here: `https://github.com/XENON1T/XeAnalysisScripts/tree/master/PaxProcessingHelpers/DumpX100Events`


How do I use pax to reduce raw data files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


How do I reduce the file size of my processed data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default we store a lot of low-level information in the processed output files. If you need smaller files, first try to make 'light' files using the reclassify configuration:

    paxer --config reclassify --input your_large_file.hdf5

This will remove fields like the per-peak sum-waveform and hitpattern from the file, reducing the filesize significantly. You can remove more or less fields by playing with the fields_to_ignore option (see light_output.ini). Whatever you do with this field, put either `all_hits` or `hits` on it: `'hits'`  is a peak property which stores all the hits in a peak, `all_hits` is an event property which stores all hits. You don't want both, and in fact you will get an error if you try.

If the files are still too big for you, try using a flattener (see XeAnalysisScripts, or write your own) to save only the main S1/S2 information. Or just select only events you need. Or just buy more disk space.



How do I use pax to generate XED files for Xerawdp processing?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pax has an XED output plugin which you can use just like other output plugins. For example, to make an XED file containing simulated events, do `paxer --config XENON100 Simulation to_XED`.

The hard part is getting Xerawdp to read the XED file you produced. For a single XED file, here is a solution that works (or at least used to):

* Make some folder on xecluster to contain everything.
* Make a subfolder `raw`, containing another subfolder `xe100_150726_1253` (I will keep using this dataset name, but you can of course put any date and time you want).
* In the `xe100_150726_1253` folder, put the XED file generated by pax. Rename it to `xe100_150726_1253.xed`.
* In the original folder, place the file `xed_test.xml` from pax's `examples`. Edit it to replace any occurrence of `/home/aalbers/xed_xdp_test` with the absolute path to your folder. 
* Ssh to `xecluster03`, then run `xerawdp -w xed_test.xml xe100_150726_1253`. The ROOT file will appear at `./processed/xe100_150726_1253/v0.4.5/xe100_150726_1253.root`.
* If you'd like to output the waveform of event 0 to .C instead, use `xerawdp -p -o xed_test.xml xe100_150726_1253 0`. The .C will appear in the current directory and can be opened by ROOT.

At the moment our hacked XML only works for one XED file (which can contain an arbitrary number of events though), and the instructions aren't very convenient. You're welcome to improve the situation!
