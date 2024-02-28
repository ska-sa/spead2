.. spead2 documentation master file, created by
   sphinx-quickstart on Mon Mar 30 09:28:58 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

spead2: high-performance data transfer
======================================

spead2 is a high-performance implementation of the :download:`SPEAD
<SPEAD_Protocol_Rev1_2012.pdf>` protocol, with both Python and C++ bindings.
The *2* in the name indicates that this is a new library, compared to the
original (and no longer maintained) PySPEAD_ library. It implements version 4
of the SPEAD protocol.

While SPEAD stands for "Streaming Protocol for Exchange of Astronomical Data",
it is not specific to astronomy. It is a generic protocol for transmitting
arrays of (mostly) numeric data. While it can be used over any packet-based or
stream-based transport protocol, it is most commonly deployed over UDP. SPEAD
over UDP might be suitable for your application if

- the data rate is fixed, such as when it is being produced by some physical
  sampling process;
- the data is largely uniform e.g. rectangular arrays of numeric data whose
  shape stays constant;
- occasional packet loss is acceptable;
- you want to multicast the data to multiple subscribers;
- you need high performance (100 Gbps+).

.. _PySPEAD: https://github.com/ska-sa/PySPEAD/

.. toctree::
   :maxdepth: 2

   installation
   tutorial/tutorial
   py
   cpp
   advanced
   perf
   tools
   migrate-3
   migrate-4
   developer
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
