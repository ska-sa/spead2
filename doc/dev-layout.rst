Repository layout
=================

C++
---

:file:`src/*.cpp`
  Source files. Files are grouped into functionality by prefixes:

  :file:`src/recv_*.cpp`
    Receiving data (``spead2::recv`` namespace).

  :file:`src/send_*.cpp`
    Sending data (``spead2::send`` namespace).

  :file:`src/common_*.cpp`
    Other general shared code (``spead`` namespace).

  :file:`src/unittest_*.cpp`
    C++ unit tests.

  :file:`src/spead2_*.cpp` and :file:`src/mcdump.cpp`
    Command-line utilities.

  :file:`src/py_*.cpp`
    Python bindings.

:file:`src/*.h`
  Header files that are only used internally (not installed for users).

:file:`include/spead/*.h`
  Header files that are installed and form the public API. The filenames
  mostly correspond to the source files.

:file:`examples/*.cpp`
  Example code.

Python
------

:file:`src/spead2/`
  Source code. This is placed within a :file:`src` subdirectory so that Python
  does not automatically import from it unless explicitly added to the Python
  path. See `Packaging a Python Library <python-src-layout_>`_ for an
  explanation of the advantages.

:file:`src/spead2/tools/`
  Implementations of the command-line tools.

:file:`examples/*.py`
  Example code.

:file:`tests/`
  Unit tests. These are mainly for use with pytest, but
  :file:`tests/shutdown.py` contains tests that are run to ensure that the
  interpret shuts down cleanly (see :ref:`interpreter-shutdown`).

Other
-----

:file:`gen/`
  Utilities that run as part of the build.

:file:`doc/`
  Documentation.

.. _python-src-layout: https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure
