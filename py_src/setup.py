"""
Setup script to build C++ pybind11 extension for SAFFRON and MLGT
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import pybind11

class get_pybind_include:
    def __str__(self):
        return pybind11.get_include()

# Custom build_ext to generate simple .so filename
class build_ext_custom(build_ext):
    def get_ext_filename(self, ext_name):
        # Return simple .so name instead of platform-specific one
        return f"{ext_name}.so"

ext_modules = [
    Extension(
        'mlgt_saffron',
        ['mlgt_saffron.cpp'],
        include_dirs=[
            get_pybind_include(), # type: ignore
            os.path.join(os.path.dirname(__file__), '..', 'assets', 'Eigen'),
            os.path.join(os.path.dirname(__file__), '..', 'assets'),
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],
    ),
]

setup(
    name='mlgt_saffron',
    version='1.0.0',
    author='Tejas Sharma',
    description='C++ extension for SAFFRON Group Testing and MLGT',
    ext_modules=ext_modules,
    install_requires=['pybind11'],
    cmdclass={'build_ext': build_ext_custom},
    zip_safe=False,
)
