Introduction to spead2
======================
spead2 is an implementation of the SPEAD_ protocol, with both Python and C++
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
- supports asynchronous operation, using trollius_ or asyncio_.

.. _SPEAD: https://casper.berkeley.edu/wiki/SPEAD
.. _PySPEAD: https://github.com/ska-sa/PySPEAD/
.. _trollius: http://trollius.readthedocs.io/
.. _asyncio: https://docs.python.org/3/library/asyncio.html

Preparation
-----------
There is optional support for netmap_ and ibverbs_ for higher performance, and
pcap_ for reading from previously captured packet dumps. If the libraries
(including development headers) are installed, they will be detected
automatically and support for them will be included.

.. _netmap: https://github.com/luigirizzo/netmap
.. _ibverbs: https://www.openfabrics.org/downloads/libibverbs/README.html
.. _pcap: http://www.tcpdump.org/

If you are installing spead2 from a git checkout, it is first necessary to run
``./bootstrap.sh`` to prepare the configure script and related files. This
requires a Python installation with trollius-fixers, pycparser and jinja2
installed. When building from a packaged download this is not required.

High-performance usage requires larger buffer sizes than Linux allows by
default. The following commands will increase the permitted buffer sizes on
Linux::

    sysctl net.core.wmem_max=16777216
    sysctl net.core.rmem_max=16777216

Note that these commands are not persistent across reboots, and the settings
need to be stored in :file:`/etc/sysctl.conf` or :file:`/etc/sysctl.d`.

Installing spead2 for Python
----------------------------
The only Python dependencies are numpy_ and six_, and trollius_ on Python
versions below 3.4 (for 3.4-3.6 trollius can still be used, and is needed to
run the test suite).

The test suite has additional dependencies; refer to
:file:`setup.py` if you are developing spead2.
Some tests depend on PySPEAD_, but they will be skipped if it is not
installed.

There are two ways to install spead2 for Python: compiling from source and
installing a binary wheel.

.. _numpy: http://www.numpy.org
.. _six: https://pythonhosted.org/six/

Installing a binary wheel
^^^^^^^^^^^^^^^^^^^^^^^^^
As from version 1.12, binary wheels are provided on PyPI for x86-64 Linux
systems. These support all the optional features, it is now the recommended
installation method as it does not depend on a compiler, development
libraries etc. The wheels use the "manylinux2010" tag, which requires at least
:command:`pip` 19.0 to install.

Provided your system meets these requirements, just run::

    pip install spead2

Python install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^
Installing from source requires a modern C++ compiler supporting C++11 (GCC
4.8+ or Clang 3.5+) as well as Boost (including compiled libraries) and the
Python development headers. At the moment only GNU/Linux and OS X get tested
but other POSIX-like systems should work too. There are no plans to support
Windows.

Installation works with standard Python installation methods.

Installing spead2 for C++
-------------------------
spead2 requires a modern C++ compiler supporting C++11 (GCC 4.8+ or Clang 3.5+)
as well as Boost (including compiled libraries). At the moment only GNU/Linux
and OS X get tested but other POSIX-like systems should work too. There are no
plans to support Windows.

The C++ API uses the standard autoconf installation flow i.e.:

.. code-block:: sh

    ./configure [options]
    make
    make install

For generic help with configuration, see :file:`INSTALL` in the top level of
the source distribution. Optional features are autodetected by default, but can
be disabled by passing options to :program:`configure` (run ``./configure -h``
to see a list of options).

One option that may squeeze out a very small amount of extra performance is
:option:`--enable-lto` to enable link-time optimization. Up to version 1.2.0
this was enabled by default, but it has been disabled because it often needs
other compiler or OS-specific configuration to make it work. For GCC, typical
usage is

.. code-block:: sh

    ./configure --enable-lto AR=gcc-ar RANLIB=gcc-ranlib

The installation will install some benchmark tools, a static library, and the
header files. At the moment there is no intention to create a shared library,
because the ABI is not stable.
