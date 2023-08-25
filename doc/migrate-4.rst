Migrating to version 4
======================

Unlike version 3, version 4 does not make substantial changes to the spead2
API, although it does drop deprecated functionality. It
replaces the build system and increases the minimum version requirements, which
may have implications for how you install, link to or use spead2.

Removed functionality
---------------------
The following deprecated functionality has been removed:

C++
^^^
.. cpp:namespace-push:: spead2

.. list-table::
   :width: 100%
   :header-rows: 1

   * - Functionality
     - Replacement
   * - :cpp:member:`!recv::udp_ibv_reader::default_buffer_size`
     - :cpp:member:`recv::udp_ibv_config::default_buffer_size`
   * - :cpp:member:`!recv::udp_ibv_reader::default_max_poll`
     - :cpp:member:`recv::udp_ibv_config::default_max_poll`
   * - :cpp:member:`!send::udp_ibv_stream::default_buffer_size`
     - :cpp:member:`send::udp_ibv_config::default_buffer_size`
   * - :cpp:member:`!send::udp_ibv_stream::default_max_poll`
     - :cpp:member:`send::udp_ibv_config::default_max_poll`
   * - :cpp:class:`recv::udp_ibv_reader` constructors that do
       not take a :cpp:class:`recv::udp_ibv_config`
     - constructor that takes a :cpp:class:`recv::udp_ibv_config`
   * - :cpp:class:`send::udp_ibv_stream` constructor that does
       not take a :cpp:class:`send::udp_ibv_config`
     - constructor that takes a :cpp:class:`send::udp_ibv_config`
   * - :cpp:class:`send::inproc_stream` constructor
       taking a single queue
     - pass a vector containing a single queue
   * - :cpp:func:`!send::inproc_stream::get_queue`
     - :cpp:func:`send::inproc_stream::get_queues`\ ``[0]``
   * - :cpp:class:`send::tcp_stream` and :cpp:class:`send::udp_stream`
       constructors taking a single endpoint
     - pass a vector containing a single endpoint

.. cpp:namespace-pop::

Python
^^^^^^

.. list-table::
   :width: 100%
   :header-rows: 1

   * - :py:const:`!recv.Stream.DEFAULT_UDP_IBV_BUFFER_SIZE`
     - :py:const:`.recv.UdpIbvConfig.DEFAULT_BUFFER_SIZE`
   * - :py:const:`!recv.Stream.DEFAULT_UDP_IBV_MAX_SIZE`
     - :py:const:`.recv.UdpIbvConfig.DEFAULT_MAX_SIZE`
   * - :py:const:`!recv.Stream.DEFAULT_UDP_IBV_MAX_POLL`
     - :py:const:`.recv.UdpIbvConfig.DEFAULT_MAX_POLL`
   * - :py:const:`!send.UdpIbvStream.DEFAULT_BUFFER_SIZE`
     - :py:const:`.send.UdpIbvConfig.DEFAULT_BUFFER_SIZE`
   * - :py:const:`!send.UdpIbvStream.DEFAULT_MAX_POLL`
     - :py:const:`.send.UdpIbvConfig.DEFAULT_MAX_POLL`
   * - :py:meth:`.recv.Stream.add_udp_ibv_reader` overload that does not take
       a :py:class:`.recv.UdpIbvConfig`
     - Pass a :py:class:`.recv.UdpIbvConfig`
   * - :py:class:`.send.UdpIbvStream` constructors that do not take a
       :py:class:`.send.UdpIbvConfig`
     - Pass a :py:class:`.send.UdpIbvConfig`
   * - :py:class:`.send.InprocStream` constructor taking a single queue
     - Pass a list containing a single queue
   * - :py:attr:`!send.InprocStream.queue`
     - :py:attr:`.send.InprocStream.queues`\ ``[0]``
   * - :py:class:`.send.TcpStream` and :py:class:`.send.UdpStream` constructors
       taking a single hostname and port
     - Pass a list containing a single :samp:`({host}, {port})` tuple

Meson
-----
The autotools build system has been replaced by `Meson`_. This mainly affects
C++ users, as for Python this is hidden behind the Python packaging
interface. Refer to the :doc:`Introduction <introduction>` for installation
instructions.

The old build system had a number of options to adjust the build. The table
below shows corresponding Meson options:

====================================== =====================================
autotools                              meson
====================================== =====================================
``--enable-debug-symbols``             ``debug=true`` or ``buildtype=...``
``--enable-debug-log``                 ``max_log_level=debug``
``--enable-coverage``                  ``b_coverage=true``
``--disable-optimized``                ``optimization=0`` or ``buildtype=debug``
``--enable-lto``                       ``b_lto=true``
``--enable-shared``                    ``default_library=both``
``--without-program-options``          ``tools=disabled``
``--without-ibv``                      ``ibv=disabled``
``--without-mlx5dv``                   ``mlx5dv=disabled``
``--without-ibv-hw-rate-limit``        ``ibv_hw_rate_limit=disabled``
``--without-pcap``                     ``pcap=disabled``
``--without-cap``                      ``cap=disabled``
``--without-recvmmsg``                 ``recvmmsg=disabled``
``--without-sendmmsg``                 ``sendmmsg=disabled``
``--without-eventfd``                  ``eventfd=disabled``
``--without-posix-semaphores``         ``posix_semaphores=disabled``
``--without-pthread_setaffinity_np``   ``pthread_setaffinity_np=disabled``
``--without-fmv``                      ``fmv=disabled``
``--without-movntdq``                  ``movntdq=disabled``
``--without-cuda``                     ``cuda=disabled``
``--without-gdrapi``                   ``gdrapi=disabled``
====================================== =====================================

Link-time optimization no longer requires intervention to select suitable
versions of :command:`ar` and :command:`ranlib`; Meson takes care of it.

C++17
-----
The codebase now uses C++17, whereas older versions used C++11. This might
require a newer C++ compiler. See the :doc:`Introduction <introduction>` for
minimum compiler versions.

Additionally, when compiling against the C++ API, you may need to pass
compiler arguments to select at least C++17 (e.g. :option:`!--std=c++17`). GCC
11+ and Clang 16+ support C++17 without a compiler flag, but keep in mind that
your users might use older compilers.

Boost
-----
Boost 1.69+ is now required: from this release, boost_system is
a header-only library. You no longer need to link against any Boost libraries
when linking against spead2.

pcap
----
The detection logic for libpcap has changed. It used to first try
:command:`pkg-config`, then fall back to testing compilation. It now tries
:command:`pkg-config` first and falls back to :command:`pcap-config`. If
neither of those methods works, you may need to upgrade your pcap library.

Code generation
---------------
In older versions of spead2, some of the code was generated and included in
the release tarballs. If you used a release, you would be unaware of this, but
trying to build directly from git would require you to run a ``bootstrap.sh``
script.

Meson doesn't have good support for including generated code into releases, so
these generated files are no longer included in the releases, and they are
instead created as part of the build. This requires Python, with the
:mod:`jinja2`, :mod:`pycparser` and :mod:`packaging` packaging installed.

An advantage of this approach is that it is now possible to directly build
from a git checkout without any preparatory steps.

Python configuration
--------------------
When building the Python bindings from source, it was previously only possible
to adjust the build-time configuration by editing source files. With
the new build system, it's now possible to `pass options`_ on the command
line.

.. _pass options: https://meson-python.readthedocs.io/en/latest/how-to-guides/config-settings.html

Python editable installs
------------------------
Meson-python `doesn't support <no-editable_>`_ editable installs with build
isolation. To make an editable install, use ``pip install --no-build-isolation -e .``.

.. _no-editable: https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html
