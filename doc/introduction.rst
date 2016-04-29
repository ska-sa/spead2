Introduction to spead2
======================

spead2 is an implementation of the SPEAD_ protocol, with both Python and C++
bindings. The *2* in the name indicates that this is a new implementation of
the protocol; the protocol remains essentially the same. Compared to the
PySPEAD_ implementation, spead2:

- is at least an order of magnitude faster when dealing with large heaps;
- correctly implements several aspects of the protocol that were implemented
  incorrectly in PySPEAD (bug-compatibility is also available);
- correctly implements many corner cases that PySPEAD would simply fail on;
- cleanly supports several SPEAD flavours (e.g. 64-40 and 64-48) in one
  module, with the receiver adapting to the flavour used by the sender;
- supports Python 3;
- supports asynchronous operation, using trollius_.

.. _SPEAD: https://casper.berkeley.edu/wiki/SPEAD
.. _PySPEAD: https://github.com/ska-sa/PySPEAD/
.. _trollius: http://trollius.readthedocs.io/

Installing spead2 for Python
----------------------------

spead2 requires a modern C++ compiler supporting C++11 (currently only GCC 4.8
and Clang 3.4 have been tried), Python development headers, and
Boost. At the moment only GNU/Linux has been tested but other POSIX-like
systems should work too.

The only Python dependencies are numpy_ and six_. Running the test suite additionally
requires nose_ and decorator_, and some tests depend on PySPEAD_ (they will be
skipped if it is not installed). Finally, the asynchronous I/O support requires trollius_.
To install (which will automatically pull in the mandatory dependencies),
run::

    ./setup.py install

Other standard methods for installing Python packages should work too.

.. _numpy: http://www.numpy.org
.. _six: https://pythonhosted.org/six/
.. _nose: https://nose.readthedocs.io/en/latest/
.. _decorator: http://pythonhosted.org//decorator/

High-performance usage requires larger buffer sizes than Linux allows by
default. The following commands will increase the permitted buffer sizes::

    sysctl net.core.wmem_max=16777216
    sysctl net.core.rmem_max=16777216

Installing spead2 for C++
-------------------------
At the moment there is no intention to create a shared library, because the
ABI is not stable. Instead, use the source files directly in your code, or
build a static library with your preferred options. The provided Makefile
produces an optimised static library.

There is optional support for :doc:`netmap <cpp-netmap>` (disabled by default)
and for acceleration using :manpage:`recvmmsg(2)` and :manpage:`eventfd(2)`
(enabled if a sufficiently new glibc is detected). To override the defaults,
pass :makevar:`NETMAP=1`, :makevar:`RECVMMSG=0` or :makevar:`EVENTFD=0` to
:program:`make`.
