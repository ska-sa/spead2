Sending
=======

Heaps
-----

.. doxygenclass:: spead2::send::heap
   :members:

.. doxygenstruct:: spead2::send::item
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

.. doxygenclass:: spead2::send::inproc_stream
   :members: inproc_stream

.. doxygenclass:: spead2::send::streambuf_stream
   :members: streambuf_stream
