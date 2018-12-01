#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:16:34 2018
Python DML Setup.
@author: jlsuarezdiaz
"""

from setuptools import setup, find_packages

from codecs import open
from os import path

from Cython.Build import cythonize

import numpy

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    #  Project name.
    #  $ pip install pyDML
    name='pyDML',

    # Version
    version='0.1.0',

    # Description
    description='Distance Metric Learning algorithms for Python',

    # Long description (README)
    long_description=long_description,

    # URL
    url='https://github.com/jlsuarezdiaz/pyDML',

    # Author
    author='Juan Luis Suárez Díaz',

    # Author email
    author_email='jlsuarezdiaz@ugr.es',

    # Classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Natural Language :: English',

        'Operating System :: OS Independent',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',

    ],

    # Keywords
    keywords=['Distance Metric Learning',
              'Classification',
              'Mahalanobis Distance',
              'Dimensionality',
              'Similarity'],

    # Packages
    packages=find_packages(exclude=['misc', 'data', 'test', 'utils', 'old']),

    # Test suite
    test_suite='test',

    # Requeriments
    install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'Cython', 'scipy', 'seaborn'],
    extras_require={},

    # Additional data
    package_data={},

    # Project urls
    project_urls={
        'Bug Reports': 'https://github.com/jlsuarezdiaz/pyDML/issues',
        # 'Funding': 'https://github.com/jlsuarezdiaz/pyDML',
        'Say Thanks!': 'https://saythanks.io/to/jlsuarezdiaz',
        # 'Contact': 'mailto:jlsuarezdiaz@ugr.es',
        'Source': 'https://github.com/jlsuarezdiaz/pyDML',
        'Documentation': 'https://pydml.readthedocs.io/',
        'Software Stats': 'https://jlsuarezdiaz.github.io/software/pyDML/stats/',
    },

    long_description_content_type='text/markdown',

    ext_modules=cythonize(["dml/*.pyx"], language="c++"),

    include_dirs=[numpy.get_include()]

)
