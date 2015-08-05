============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/XENON1T/pax/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
is open to whoever wants to implement it.  There is also a "help wanted" flag that 
indicates what is open to anybody to help.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "feature"
is open to whoever wants to implement it, if no other name is assigned.

Write Documentation
~~~~~~~~~~~~~~~~~~~

`pax` could always use more documentation, whether as part of the official pax docs, in docstrings, or even on the wiki.  The documentation is automatically generated with sphinx.  You can generate the docs by going into the docs folder and running::

    $ make html
    
Make sure to have updated Sphinx by::
    
    $ pip install -U sphinx
    
Feel free to expand the documentation however you find helpful.

Now you can also update the documentation on the site if needed. Go to the folder where pax is cloned and run::
    
    $ git clone git@github.com:XENON1T/pax.git paxdocs
    $ cd paxdocs
    $ git pull
    $ git checkout gh-pages
    

You are now ready to make the docs for the site. Go into the pax directory and run::
    
    $ make docs
       

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/XENON1T/pax/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

Get Started with developing!
----------------------------

Ready to contribute? Here's how to set up `pax` for local development.  If you get lost, feel free to ask the developers for help or explore the extensive documentation that exists for github.

1. Clone locally::

    $ git clone https://github.com/XENON1T/pax.git

2. Install your local copy.  (If you chose not to use Anaconda, you may want to
install in a virtual environment.)  This is how you set up your fork for local
development::

    $ cd pax/
    $ python setup.py develop

3. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox::

    $ make lint
    $ make test

   This runs flake8 for style testing.  To get flake8, just pip install them into your virtualenv.
   
5.  Check that the documentation is still up-to-date.  You may need to add another 'rst' file in docs or
update one of the files that are already written.  Please specifically check README.rst (for new features)
and AUTHORS.rst (if you are a new contributor)

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check the following:

- Have you accidentally modified unrelated files?
- Please include at least basic documentation of new features (such as docstrings for functions and classes).
- Please include unit tests (see the `/tests` folder) for new features. This will greatly help speed up the review process, as we can already be sure your code does what it is supposed to do.
- Please ensure the other tests of pax still pass (try `python setup.py test`).
- New code should by PEP8-style compliant, although line lengths are allowed to extend up to 120 characters. To check this, use `flake8 --max-line-length 120 pax tests bin`. We resolve style arguments by fights to the death. Whatever you do, don't repeat this http://imgs.xkcd.com/comics/code_quality.png.

Your pull request will be reviewed by somebody who is not you. If you get asked to do a review, we're asking you to do a 30 minutes to 1 hour check:

- Are any unrelated files accidentally modified?
- Does it actually work? For most PRs there should be a few new unit tests defined, which are run automatically by Travis (which gives the green checkmark). 
- Do you think this is a change that belongs in pax?
- Is the documentation (text in the pull request, docstrings and comments in code) sufficient to figure out roughly what's going on?
 
If you are completely satisfied, and the Travis CI build passes, you can merge the pull request immediately; else you can ask for a few improvements or recommend something else.

For very small changes, you may commit directly to the master branch if you are already
a frequent contributor or maintainer.  If your change is actually not small or you aren't
a frequent commiter, your commit may be undone.

If your change is large,  please consider breaking up your changes into smaller increments.
If this is not possible, please contact the developers via an issue to discuss how to proceed.
It is just really difficult to manage a project if major changes come unexpectedly.  Therefore,
we recommend you commit early and often.

Tips
----

To run a subset of tests::

	$ python -m unittest tests.test_pax
