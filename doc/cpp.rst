C++ API for spead2
==================

The C++ API is at a lower level than the Python API. In particular, item
values are treated as uninterpreted binary blobs. The protocol is directly
tied to numpy's type system, so it is not practical to implement this in C++.
The C++ API is thus best suited to situations which require the maximum
possible performance and where the data formats can be fixed in advance.

There is also no equivalent to the :py:class:`spead2.ItemGroup` and
:py:class:`spead2.send.HeapGenerator` classes. The user is responsible for
maintaining previously seen descriptors (if they are desired) and tracking
which descriptors and items need to be inserted into heaps.

The C++ documentation is far from complete. As a first step, consult the Python
documentation; in many cases it is just wrapping the C++ interface with Pythonic
names, whereas the C++ interface uses lowercase with underscores for
all names. If that doesn't help, consult the Doxygen-style comments in the
source code.

The compiler and link flags necessary for compiling and linking against spead2
can be found with :program:`pkg-config` i.e.,

 - ``pkg-config --cflags spead2`` to get the compiler flags
 - ``pkg-config --libs --static spead2`` to get the linker flags

Note that when installed with the default setup on a GNU/Linux system, the
:file:`spead2.pc` file is installed outside :program:`pkg-config`'s default
search path, and you need to set :envvar:`PKG_CONFIG_PATH` to
:file:`/usr/local/lib/pkgconfig` first.

.. toctree::
   :maxdepth: 2

   cpp-stability
   cpp-asio
   cpp-recv
   cpp-send
   cpp-inproc
   cpp-logging
   cpp-ibverbs
   cpp-netmap
