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
- supports asynchronous operation, using trollius_.

.. _SPEAD: https://casper.berkeley.edu/wiki/SPEAD
.. _PySPEAD: https://github.com/ska-sa/PySPEAD/
.. _trollius: http://trollius.readthedocs.io/

Preparation
-----------
spead2 requires a modern C++ compiler supporting C++11 (currently only GCC 4.8
and Clang 3.4 have been tried) as well as Boost (including compiled libraries).
The Python bindings have additional dependencies — see below. At the moment
only GNU/Linux has been tested but other POSIX-like systems should work too (OS
X is tested occasionally).

There is optional support for netmap_ and ibverbs_ for higher performance. If
the libraries (including development headers) libraries are installed, they
will automatically be detected and used.

.. _netmap: https://github.com/luigirizzo/netmap
.. _ibverbs: https://www.openfabrics.org/downloads/libibverbs/README.html

If you are installing spead2 from a git checkout, it is first necessary to run
``./bootstrap.sh`` to prepare the configure script and related files. When
building from a packaged download this is not required.

High-performance usage requires larger buffer sizes than Linux allows by
default. The following commands will increase the permitted buffer sizes on
Linux::

    sysctl net.core.wmem_max=16777216
    sysctl net.core.rmem_max=16777216

Note that these commands are not persistent across reboots, and the settings
need to be stored in :file:`/etc/sysctl.conf` or :file:`/etc/sysctl.d`.

Installing spead2 for Python
----------------------------
The only Python dependencies are numpy_ and six_, although support for
asynchronous I/O also requires trollius_. Running the test suite additionally
requires nose_, decorator_ and netifaces_, and some tests depend on PySPEAD_
(they will be skipped if it is not installed). It is also necessary to have the
development headers for Python, and Boost.Python.

To install (which will automatically pull in the mandatory dependencies), run::

    ./setup.py install

Other standard methods for installing Python packages should work too.

.. _numpy: http://www.numpy.org
.. _six: https://pythonhosted.org/six/
.. _nose: https://nose.readthedocs.io/en/latest/
.. _decorator: http://pythonhosted.org/decorator/
.. _netifaces: https://pypi.python.org/pypi/netifaces

Installing spead2 for C++
-------------------------
The C++ API uses the standard autoconf installation flow i.e.:

.. code-block:: sh

    ./configure [options]
    make
    make install

For generic help with configuration, see :file:`INSTALL` in the top level of
the source distribution. Optional features are autodetected by default, but can
be disabled by passing options to :program:`configure` (run ``./configure -h``
to see a list of options).

The installation will install some benchmark tools, a static library, and the
header files. At the moment there is no intention to create a shared library,
because the ABI is not stable.
