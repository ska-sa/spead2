Asynchronous I/O
================
The C++ API uses Boost.Asio for asynchronous operations. There is a
:cpp:class:`spead2::thread_pool` class (essentially the same as the Python
:py:class:`spead2.ThreadPool` class). However, it is not
required to use this, and you may for example run everything in one thread to
avoid multi-threading issues.

.. doxygenclass:: spead2::thread_pool
   :members:

A number of the APIs use callbacks. These follow the usual Boost.Asio
guarantee that they will always be called from threads running
:cpp:func:`boost::asio::io_service::run`. If using a
:cpp:class:`~spead2::thread_pool`, this will be one of the threads managed by
the pool. Additionally, callbacks for a specific stream are serialised, but
there may be concurrent callbacks associated with different streams.
