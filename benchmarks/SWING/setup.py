from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md')


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='Swing',
    url='https://github.com/bagherilab/SWING',
    license='Apache Software License',
    author='Justin Finkle & Jia Wu',
    tests_require=['pytest'],
    install_requires=['numpy', 'scipy','scikit-learn', 'pandas'],
    cmdclass={'test': PyTest},
    author_email='jfinkle@u.northwestern.edu',
    description='Sliding window inference methods',
    long_description=long_description,
    #packages=['Swing'],
    packages=find_packages(include=["Swing", "Swing.*"]), 
    include_package_data=True,
    platforms='any',
    classifiers=[],
    extras_require={
        'testing': ['pytest'],
    }
)
