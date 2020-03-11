from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules =[Extension("Scroll", ["Scroll.pyx"],
                        include_dirs=[numpy.get_include()],
                        extra_compile_args=['/openmp'],
                        extra_link_args=['/openmp'])
              ]

setup(
  name="Scroll",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules,
  include_dirs=[numpy.get_include()]
)
