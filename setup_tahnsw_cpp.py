"""
setup_tahnsw_cpp.py
Build the patched hnswlib as 'tahnsw_cpp' Python extension.

Usage:
    python setup_tahnsw_cpp.py build_ext --inplace
"""
import os
import sys
from setuptools import setup, Extension
import pybind11

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src", "hnswlib")

ext = Extension(
    "tahnsw_cpp",
    sources=[os.path.join(SRC, "python_bindings", "bindings.cpp")],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(SRC, "hnswlib"),   # tahnsw_alg.h + topology_math.h
        SRC,                            # hnswlib root (hnswlib.h etc.)
    ],
    extra_compile_args=[
        "-O3", "-march=native", "-std=c++17",
        "-ffast-math",
        # silence hnswlib warnings that clutter the build output
        "-Wno-sign-compare", "-Wno-unused-variable",
    ],
    language="c++",
)

setup(
    name="tahnsw_cpp",
    version="0.1.0",
    ext_modules=[ext],
)
