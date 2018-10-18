#MIT License
#
#Copyright (c) 2018 Dafiti OpenSource
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

# We used setup.py from the requests library as reference:
# https://github.com/requests/requests/blob/master/setup.py
import os
import re
import sys

from codecs import open

from setuptools import setup
from setuptools.command.test import test as TestCommand

here = os.path.abspath(os.path.dirname(__file__))

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist bdist_wheel')
    os.system('twine upload dist/*')
    sys.exit()

install_requires = [
    'statsmodels>=0.9.0',
    'matplotlib',
    'jinja2'
]

test_requires = [
    'pytest',
    'pytest-coverage',
    'flake8',
    'isort',
    'tox'
]

packages = ['causalimpact']

_version = {}
_version_path = os.path.join(here, 'causalimpact', '__version__.py')
with open(_version_path, 'r', 'utf-8') as f:
    exec(f.read(), _version)

with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name='pycausalimpact',
    version=_version['__version__'],
    author='Willian Fuks',
    author_email='willian.fuks@gmail.com',
    url='https://github.com/dafiti/causalimpact',
    description= "Python version of Google's Causal Impact model",
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    test_requires=test_requires,
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
    ]
)
