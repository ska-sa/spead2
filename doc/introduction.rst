Introduction to spead2
======================
spead2 is an implementation of the :download:`SPEAD <SPEAD_Protocol_Rev1_2012.pdf>`
protocol, with both Python and C++
bindings. The *2* in the name indicates that this is a new implementation of
the protocol; the protocol remains essentially the same. Compared to the
PySPEAD_ implementation, spead2:

- is at least an order of magnitude faster when dealing with large heaps;
- correctly implements several aspects of the protocol that were implemented
  incorrectly in PySPEAD (bug-compatibility is also available);
- correctly implements many corner cases on which PySPEAD would simply fail;
- cleanly supports several SPEAD flavours (e.g. 64-40 and 64-48) in one
  module, with the receiver adapting to the flavour used by the sender;
- supports Python 3;
- supports asynchronous operation, using asyncio_.

.. _PySPEAD: https://github.com/ska-sa/PySPEAD/
.. _asyncio: https://docs.python.org/3/library/asyncio.html

Preparation
-----------
There is optional support for ibverbs_ for higher performance, and
pcap_ for reading from previously captured packet dumps. If the libraries
(including development headers) are installed, they will be detected
automatically and support for them will be included.

.. _ibverbs: https://www.openfabrics.org/downloads/libibverbs/README.html
.. _pcap: http://www.tcpdump.org/

High-performance usage requires larger buffer sizes than Linux allows by
default. The following commands will increase the permitted buffer sizes on
Linux::

    sysctl net.core.wmem_max=16777216
    sysctl net.core.rmem_max=16777216

Note that these commands are not persistent across reboots, and the settings
need to be stored in :file:`/etc/sysctl.conf` or :file:`/etc/sysctl.d`.

Installing spead2 for Python
----------------------------
The only Python dependency is numpy_.

The test suite has additional dependencies; refer to :doc:`dev-setup`
if you are developing spead2.

There are two ways to install spead2 for Python: compiling from source and
installing a binary wheel.

.. _numpy: http://www.numpy.org

Installing a binary wheel
^^^^^^^^^^^^^^^^^^^^^^^^^
As from version 1.12, binary wheels are provided on PyPI for x86-64 Linux
systems. These support all the optional features, and it is the recommended
installation method as it does not depend on a compiler, development
libraries etc. The wheels use the "manylinux2014" tag, which requires at least
:command:`pip` 19.3 to install.

Since version 4.0 there are also aarch64 Linux wheels, which use the
"manylinux_2_28" tag and require at least :command:`pip` 20.3; and wheels for
MacOS (both Intel and Apple Silicon).

Provided your system meets these requirements, just run::

    pip install spead2

.. _py-install-source:

Python install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing from source requires a modern C++ compiler supporting C++17 (GCC
7+ or Clang 4+, although GCC 9.4 and Clang 10 are the oldest tested
versions and support for older compilers may be dropped) as well as Boost 1.69+
(only headers are required), libdivide, and the Python development headers.
At the moment only GNU/Linux and OS X get tested but other POSIX-like systems
should work too. There are no plans to support Windows.

Installation works with standard Python installation methods.

.. _cxx-install:

Installing spead2 for C++
-------------------------
Installing spead2 requires

- a modern C++ compiler supporting C++17 (see above for supported compilers)
- Boost 1.69+, including the compiled boost_program_options library
- libdivide
- Python 3.x, with the packaging, jinja2, and pycparser packages
- `Meson`_ 1.2 or later (note that this might be newer than the Meson package
  in your operating system's package manager).

.. _Meson: https://mesonbuild.com/

At the moment only GNU/Linux and OS X get tested but
other POSIX-like systems should work too. There are no plans to support
Windows.

Compilation uses the standard Meson flow (refer to the Meson manual for further
help):

.. code-block:: sh

    meson setup [options] build
    cd build
    meson compile
    meson install

Optional features are autodetected by default, but can be disabled using
Meson options. To see the available options, run :command:`meson configure` in
the build directory.
One option that may squeeze out a very small amount of extra performance is
link-time optimization, enabled with :option:`!-Db_lto=true`.

The installation will install some benchmark tools, a static library, and the
header files.

Shared library
^^^^^^^^^^^^^^
There is experimental support for building a shared library. Pass
``--default_library=both`` to ``meson setup``. It's also possible to pass
``--default_library=shared``, in which case the static library will not be
built, and the command-line tools will be linked against the shared library.

It's not recommended for general use because the binary interface is likely to
be incompatible between spead2 versions, requiring software linked against the
shared library to be recompiled after upgrading spead2 (which defeats one of
the points of a shared library). It also exports a lot of symbols (e.g., from
Boost) that may clash with other libraries. Performance may be lower than using
the static library. It is made available for users who need to load the
library dynamically as part of a plugin system.
