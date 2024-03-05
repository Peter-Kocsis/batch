"""
A setuptools based setup module. Based on https://github.com/ultralytics/pip?tab=readme-ov-file

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib
import re

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()  # current path
long_description = (here / "README.md").read_text(encoding="utf-8")  # Get the long description from the README file


def get_version():
    file = here / "src/batch/__init__.py"
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(), re.M).group(1)


setup(
    name="batch-dev",  # Required https://packaging.python.org/specifications/core-metadata/#name
    version=get_version(),  # Required https://packaging.python.org/en/latest/single_source_version.html
    description="Generic python module for handling dictionary-based batch data",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional
    url="https://github.com/Peter-Kocsis/batch",  # Optional, project's main homepage
    author="Peter Kocsis",  # Optional, name or the name of the organization which owns the project
    author_email="peti0510@gmail.com",  # Optional
    classifiers=[
        "Development Status :: 4 - Beta",  # 3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Intended Audience :: Developers",  # Indicate who your project is intended for
        "Operating System :: OS Independent",
        "Topic :: Education",  # Topics
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",  # Pick your license as you wish
        "Programming Language :: Python :: 3.8",  # Python version support
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],  # Classifiers help users find your project by categorizing it https://pypi.org/classifiers/
    keywords="data-processing, datastructure, machine-learning, deep-learning, ml, pytorch, numpy",  # Optional
    package_dir={"": "src"},  # Optional, use if source code is in a subdirectory under the project root, i.e. `src/`
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.8, <4",
    project_urls={
        "Bug Reports": "https://github.com/Peter-Kocsis/batch/issues",
        "Source": "https://github.com/Peter-Kocsis/batch",
    },  # Optional https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
)


