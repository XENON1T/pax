.PHONY: clean-pyc clean-build docs clean

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation and upload to Github"
	@echo "major - tag, push, package and upload a major release"
	@echo "minor - tag, push, package and upload a minor release"
	@echo "patch - tag, push, package and upload a patch release"

clean: clean-build clean-pyc
	rm -fr htmlcov/
	rm -f test_tree.root output1.hdf5 output2.hdf5 xe100_120402_2000_000000_pax4.0.1.root xe100_120402_2000_000000_pax4.0.1.hdf5

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint:
	flake8 --max-line-length=120 pax tests bin

test:	clean
	python setup.py test

test-all:
	tox

coverage:
	pip install -U nose
	nosetests --with-coverage --cover-package=pax tests -e test_root

docs:

	# For this to work, you have to first:
	#  cd ..
	#  git clone  git@github.com:XENON1T/pax.git paxdocs
	#  cd paxdocs
	#  git checkout --orphan gh-pages

	rm -f docs/pax.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pax
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

	cp -r docs/_build/html/* ../paxdocs/
	bash -c "cd ../paxdocs;git add -A;git commit -m \"Generated gh-pages\";git push origin gh-pages;cd ../pax"

	echo open docs/_build/html/index.html

major: clean docs
	vim HISTORY.rst
	git commit -m "Update HISTORY for the release" HISTORY.rst
	bumpversion major
	git push --tags
	git push
	git checkout stable
	git merge master
	git push --tags
	git push
	git checkout master

minor: clean docs
	vim HISTORY.rst
	git commit -m "Update HISTORY for the release" HISTORY.rst
	bumpversion minor
	git push --tags
	git push
	git checkout stable
	git merge master
	git push --tags
	git push
	git checkout master

patch: clean docs
	vim HISTORY.rst
	git commit -m "Update HISTORY for the release" HISTORY.rst
	bumpversion patch
	git push --tags
	git push
	git checkout stable
	git merge master
	git push --tags
	git push
	git checkout master
