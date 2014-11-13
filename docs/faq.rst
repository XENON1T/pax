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