.PHONY: clean-pyc clean-build docs clean

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "major - tag, push, package and upload a major release"
	@echo "minor - tag, push, package and upload a minor release"
	@echo "patch - tag, push, package and upload a patch release"

clean: clean-build clean-pyc
	rm -fr htmlcov/

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint:
	flake8 --max-line-length=100 pax tests

test:
	python setup.py test

test-all:
	tox

coverage:
	coverage run --source pax setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/pax.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pax
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/_build/html/index.html

major: clean docs
	bumpversion major
	emacs HISTORY.rst
	git commit -m "Update HISTORY for the release" HISTORY.rst
	git push
	git push --tags


minor: clean docs
	bumpversion minor
	emacs HISTORY.rst
	git commit -m "Update HISTORY for the release" HISTORY.rst
	git push
	git push --tags

patch: clean docs
	bumpversion patch
	emacs HISTORY.rst
	git commit -m "Update HISTORY for the release" HISTORY.rst
	git push
	git push --tags
