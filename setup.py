#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

requirements = open('requirements.txt').read().splitlines()

test_requirements = requirements + ['flake8',
                                    'tox',
                                    'coverage',
                                    'bumpversion']

setup(
    name='pax',
    version='1.4.0',
    description='PAX is the raw data processor for the XENON1T experiment, with support for other LXe TPCs.',
    long_description=readme + '\n\n' + history,
    author='Christopher Tunnell and Jelle Aalbers for the XENON1T collaboration',
    author_email='ctunnell@nikhef.nl',
    url='https://github.com/XENON1T/pax',
    packages=[
        'pax',
        'pax.config',
        'pax.plugins',
        'pax.plugins.corrections',
        'pax.plugins.signal_processing',
        'pax.plugins.for_testing',
        'pax.plugins.io',
        'pax.plugins.posrec',
        'pax.micromodels'
    ],
    package_dir={'pax': 'pax'},
    package_data={'pax': ['config/*.ini', 'data/*.*']},
    scripts=['bin/paxer'],
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='pax',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
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
