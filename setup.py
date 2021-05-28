# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import setuptools
import inspect
import sys
import os

long_description = """Qiskit Machine Learning is a open-source library of quantum computing machine learning experiments.
 """

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

if not hasattr(setuptools, 'find_namespace_packages') or not inspect.ismethod(setuptools.find_namespace_packages):
    print("Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
          "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__))
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit_machine_learning", "VERSION.txt")
with open(VERSION_PATH, "r") as version_file:
    VERSION = version_file.read().strip()

setuptools.setup(
    name='qiskit-machine-learning',
    version=VERSION,
    description='Qiskit Machine Learning: A library of quantum computing machine learning experiments',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Qiskit/qiskit-machine-learning',
    author='Qiskit Machine Learning Development Team',
    author_email='hello@qiskit.org',
    license='Apache-2.0',
    classifiers=(
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering"
    ),
    keywords='qiskit sdk quantum machine learning ml',
    packages=setuptools.find_packages(include=['qiskit_machine_learning','qiskit_machine_learning.*']),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.6",
    extras_require={
        'torch': ["torch"],
        'sparse': ["sparse"],
    },
    zip_safe=False
)
