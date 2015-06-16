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
    
Feel free to expand the documentation however you find helpful.

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

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
   
Every pull request will be reviewed.  This review will happen with somebody who was not
the original author of the pull request.  The person doing the reviewing will spend between 
30 minutes and one hour reviewing the code.  They will check that your new code contributes
something to the project, will be used, is tested, is clear what it does, and is documented.
Typically, there will be one round of iteration that should not take more than 1 day of work
for the person authoring the pull request.  Lastly, the person doing the review will ensure
that the Travis CI build passes.  Once the reviewer is confident that this pull request
does not result in untested code that nobody understands or uses, the reviewer does the merge.

For very small changes, you may commit directly to the master breanch if you are already
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
