import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#compile: python setup.py build_ext --inplace
filename='fast_gauss_blur.pyx'

setup(
    name=filename.split('.')[0],
    cmdclass={'build_ext':build_ext},
    ext_modules=[Extension(filename.split('.')[0],
    sources=[filename,"fgb.c"],
    include_dirs=[numpy.get_include()])],
    language='c',
)