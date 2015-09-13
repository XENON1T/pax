#!/usr/bin/env python
# -*- coding: utf-8 -*-
import six
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = open('requirements.txt').read().splitlines()

# The avro package has a different name in python2:
if six.PY2:
    del requirements[requirements.index('avro-python3')]
    requirements.append('avro')

# For some reason h5py is often not seen by pip if it was installed by conda...
# so check for h5py presence manually, and remove it from requirements if already found.
try:
    import h5py
except ImportError:
    pass
else:
    del requirements[requirements.index('h5py')]

# Snappy cannot be installed automatically on windows
if os.name == 'nt':
    print("You're on windows: we can't install snappy manually. "
          "See http://xenon1t.github.io/pax/faq.html#can-i-set-up-pax-on-my-windows-machine")
    del requirements[requirements.index('python-snappy>=0.5')]

test_requirements = requirements + ['flake8',
                                    'tox',
                                    'coverage',
                                    'bumpversion']

setup(
    name='pax',
    version='3.3.0',
    description='PAX is the raw data processor for the XENON1T experiment, with support for other LXe TPCs.',
    long_description=readme + '\n\n' + history,
    author='Christopher Tunnell and Jelle Aalbers for the XENON1T collaboration',
    author_email='ctunnell@nikhef.nl',
    url='https://github.com/XENON1T/pax',
    packages=[  'pax',
                'pax.config',
                'pax.plugins',
                'pax.plugins.io',
                'pax.plugins.signal_processing',
                'pax.plugins.peak_processing',
                'pax.plugins.posrec',
                'pax.plugins.interaction_processing',
                ],
    package_dir={'pax': 'pax'},
    package_data={'pax': ['config/*.ini', 'data/*.*']},
    scripts=['bin/paxer', 'bin/event-builder',
             'bin/convert_pax_formats'],
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='pax',
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
