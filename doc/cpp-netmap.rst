Support for netmap
==================

Introduction
------------
As an experimental feature, it is possible to use the netmap_ framework to
receive packets at a higher rate than is possible with the regular sockets
API. This is particularly useful for small packets.

.. _netmap: info.iet.unipi.it/~luigi/netmap/

This is not for the faint of heart: it requires root access, it can easily
hang the whole machine, and it imposes limitations, including:

- Only the C++ API is supported. If you need every drop of performance, you
  shouldn't be using Python anyway.
- Only Linux is currently tested. It should be theoretically possible to
  support FreeBSD, but you're on your own (patches welcome).
- Only IPv4 is supported.
- Fragmented IP packets, and IP headers with optional fields are not
  supported.
- Checksums are not validated (although possibly the NIC will check them).
- Only one reader is supported per network interface.
- All packets that arrive with the correct UDP port will be processed,
  regardless of destination address. This could mean, for example, that
  unrelated multicast streams will be processed even though they aren't
  wanted.

Usage
-----
Once netmap is installed and the header file :file:`net/netmap_user.h` is placed in
a system include directory, pass :makevar:`NETMAP=1` to :program:`make` to include netmap
support in the library.

Then, instead of :cpp:class:`spead2::recv::udp_reader`, use
:cpp:class:`spead2::recv::netmap_udp_reader`.

.. doxygenclass:: spead2::recv::netmap_udp_reader
   :members: netmap_udp_reader
