from setuptools import setup
import numpy as np
from Cython.Build import cythonize

setup(
    name='autograd_linalg',
    author='Matthew James Johnson',
    author_email='mattjj@csail.mit.edu',
    packages=['autograd_linalg'],
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),],
    install_requires=['numpy >= 1.10.0', 'cython >= 0.24', 'scipy >= 0.18.1',
                      'autograd'],
)
