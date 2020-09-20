Sending
=======

Heaps
-----

.. doxygenclass:: spead2::send::heap
   :members:

.. doxygenstruct:: spead2::send::item
   :members:

Configuration
-------------
See :py:class:`spead2.send.StreamConfig` for an explanation of the
configuration options. In the C++ API, one must first construct a default
configuration and then use setters to set individual properties. The setters
all return the configuration itself so that one can construct a configuration
with a single expression such as

.. code:: c++

   spead2::send::stream_config().set_max_packet_size(9172).set_rate(1e9)

.. doxygenclass:: spead2::send::stream_config
   :members:

Streams
-------
All stream types are derived from :cpp:class:`spead2::send::stream` using the
`curiously recurring template pattern`_ and implementing an
:samp:`async_send_packet` function.

.. _`curiously recurring template pattern`: http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

.. doxygentypedef:: spead2::send::stream::completion_handler

.. doxygenclass:: spead2::send::stream
   :members:

.. doxygenclass:: spead2::send::udp_stream
   :members: udp_stream

.. doxygenclass:: spead2::send::tcp_stream
   :members: tcp_stream

.. doxygenclass:: spead2::send::streambuf_stream
   :members: streambuf_stream
