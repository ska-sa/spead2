Asynchronous I/O
================
The C++ API uses Boost.Asio for asynchronous operations. There is a
:cpp:class:`spead2::thread_pool` class (essentially the same as the Python
:py:class:`spead2.ThreadPool` class). However, it is not
required to use this, and you may for example run everything in one thread to
avoid multi-threading issues.

.. doxygenclass:: spead2::thread_pool
   :members:

Classes that perform asynchronous operations take a parameter of type
:cpp:class:`spead2::io_service_ref`. This can be (implicitly) initialised from
either a :cpp:class:`boost::asio::io_service` reference, a
:cpp:class:`spead2::thread_pool` reference, or a
:cpp:class:`std::shared_ptr\<spead2::thread_pool\>`. In the last case, the
receiving class retains a copy of the shared pointer, providing convenient
lifetime management of a thread pool.

.. doxygenclass:: spead2::io_service_ref
   :members:

A number of the APIs use callbacks. These follow the usual Boost.Asio
guarantee that they will always be called from threads running
:cpp:func:`boost::asio::io_service::run`. If using a
:cpp:class:`~spead2::thread_pool`, this will be one of the threads managed by
the pool. Additionally, callbacks for a specific stream are serialised, but
there may be concurrent callbacks associated with different streams.
