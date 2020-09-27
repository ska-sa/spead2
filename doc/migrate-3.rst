Migrating to version 3
======================

Version 3 makes a number of breaking changes, for the purpose of keeping the
number of constructor arguments under control and making future extension more
manageable. Almost all code will need to be updated, but the updates will in
most cases be minor.

To allow for code that wishes to support both version 3 and older versions,
C++ macros are defined to allow the version number to be interrogated at
compile time. Thus, version 3 can be detected as

.. code:: c++

   #if defined(SPEAD2_MAJOR) && SPEAD2_MAJOR >= 3
   // Version 3 or later
   #else
   // Older
   #endif

Note that version 3.0.0b1 did not define these macros, but also did not
include the breaking changes.

.. c:macro:: SPEAD2_MAJOR

   Major version of spead2 e.g., ``3`` for version 3.4.6.

   .. versionadded: 3.0

.. c:macro:: SPEAD2_MINOR

   Minor version of spead2 e.g., ``4`` for version 3.4.6.

.. c:macro:: SPEAD2_PATCH

   Patch level of spead2 e.g., ``6`` for version 3.4.6.

.. c:macro:: SPEAD2_VERSION

   Full spead2 version number, as a string constant.

In Python, one can get the full version string from
:py:data:`spead2.__version__`. Use the classes in :py:mod:`distutils.version`
to analyse it.

Receive stream configuration
----------------------------
Prior to version 3, some parameters to configure a stream were passed directly
to the constructor (e.g., the maximum number of partial heaps), while others
were set by methods after construction (such as the memory allocator). In
version 3, all these parameters are set at construction time, and they are
help in helper classes :py:class:`spead2.recv.StreamConfig` and
:py:class:`spead2.recv.RingStreamConfig`
(:cpp:class:`spead2::recv::stream_config` and
:cpp:class:`spead2::recv::ring_stream_config` for C++). Code will need to be
modified to construct these helper objects.

In version 2 it was also possible (although not recommended) to change
parameters like the memory allocator after readers had already been placed.
For efficiency reasons this is no longer supported in version 3.

Send stream configuration
-------------------------
The changes for sending are more minor: the constructor for the Python class
:py:class:`spead2.send.StreamConfig` now only takes keyword arguments, and the
C++ equivalent :class:`spead2::send::stream_config` takes no constructor
arguments. To make it convenient to construct temporaries, the
setter methods return the object, allowing configurations to be constructed in
a "fluent" style e.g.:

.. code:: c++

   spead2::send::stream_config().set_max_packet_size(9172).set_rate(1e6)

Substreams
----------
A new feature is the ability to create a send stream with multiple destinations
and select the destination on a per-heap basis (see :ref:`py-substreams` for
more information). Supporting this cleanly required a number of changes:

- The :attr:`spead2.send.InprocStream.queue` attribute has been removed, and replaced with
  :attr:`~spead2.send.InprocStream.queues`. Similarly, the C++
  :cpp:func:`spead2::send::inproc_stream::get_queue` has been replaced by
  :cpp:func:`~spead2::send::inproc_stream::get_queues`.
- The constructors for most send stream types now accept a list of endpoints
  (or queues) rather than a single endpoint (queue). The old constructors are
  still supported for backwards compatibility, but are deprecated.
- The :program:`spead2_send` and :program:`spead2_send.py` example programs now
  take the destination in the form :samp:`{host}:{port}` instead of
  :samp:`{host} {port}`, and support multiple destinations.

Out-of-order packets
--------------------
In prior versions of spead2, the packets forming a single heap could be
received in any order. Starting with version 3, the default is to assume that
packets arrive in order. Refer to :ref:`py-packet-ordering` for more
details.

Loop argument to asyncio functions
----------------------------------
The Python asyncio-based classes and functions no longer take a `loop`
argument. As of Python 3.6 (which is now the minimum supported version),
:py:func:`asyncio.get_event_loop` returns the executing event loop, so there
is no need to pass the loop explicitly.

Removal of deprecated functionality
-----------------------------------
The following functions were deprecated in version 2 and have been removed in version 3:

- C++ stream constructors that specified a socket but not an :cpp:class:`!io_service`
  (they could not be supported with Boost 1.70 onwards).
- Stream constructors that took both an existing (but unconnected) socket and a
  buffer size or a port to bind to. The caller should instead bind the socket
  (if receiving) and set any desired buffer size socket option.
