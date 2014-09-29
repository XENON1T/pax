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
	version='1.1.0',
	description='PAX is used for doing digital signal processing and other data processing on the XENON1T raw data',
	long_description=readme + '\n\n' + history,
	author='Christopher Tunnell',
	author_email='ctunnell@nikhef.nl',
	url='https://github.com/XENON1T/pax',
	packages=[
		'pax',
	],
	package_dir={'pax': 'pax'},
	package_data={'pax': ['config/*.ini']},
	scripts = ['bin/paxer'],
	install_requires=requirements,
	license="BSD",
	zip_safe=False,
	keywords='pax',
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Developers',
		'License :: OSI Approved :: BSD License',
		'Natural Language :: English',
		"Programming Language :: Python :: 2",
		'Programming Language :: Python :: 2.6',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 3.3',
		'Programming Language :: Python :: 3.4',
	],
	test_suite='tests',
	tests_require=test_requirements
)
