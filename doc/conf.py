# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import subprocess
import sys

import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
sys.path.insert(0, os.path.abspath('..'))

# Build doxygen first on readthedocs
rtd = os.environ.get('READTHEDOCS') == 'True'
if rtd:
    subprocess.check_call(['doxygen'])
    # ReadTheDocs can't build the extension (no Boost), so we have to mock it
    # See http://docs.readthedocs.io/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules
    import unittest.mock
    MOCK_MODULES = ['spead2._spead2', 'spead2._spead2.send', 'spead2._spead2.recv']
    sys.modules.update((mod_name, unittest.mock.Mock()) for mod_name in MOCK_MODULES)
    # Mocking certain classes causes subclasses not to be documented properly
    sys.modules['spead2._spead2.send'].UdpStreamAsyncio = object
    sys.modules['spead2._spead2.send'].UdpIbvStreamAsyncio = object

# -- Project information -----------------------------------------------------

project = 'spead2'
copyright = '2015–2021, National Research Foundation (SARAO)'
author = 'National Research Foundation (SARAO)'

def get_version():
    globals_ = {}
    root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(root, 'src', 'spead2', '_version.py')) as f:
        code = f.read()
    exec(code, globals_)
    release = globals_['__version__']
    match = re.match('^(\d+)\.(\d+)', release)
    return match.group(0), release

version, release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'breathe'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

breathe_projects = {'spead2': './doxygen/xml'}
breathe_default_project = 'spead2'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'numba': ('https://numba.readthedocs.io/en/latest/', None)
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
