In-process transport
--------------------
While SPEAD is generally deployed over UDP, it is less than ideal for writing tests:

- One has to deal with allocating port numbers and avoiding conflicts.
- The sender and receiver need to be running at the same time.
- If the receiver doesn't keep up, it can drop packets.

To simplify unit testing, spead2 also offers an "in-process" transport. One
creates a queue, then connects a sender and a receiver to it. The queue has
unbounded capacity, so one can safely send all the data first, then create
the receiver later. This unbounded capacity also means that it should *not* be
used in production for high-volume streams, because it can exhaust all your
memory if the sender works faster than the receiver.

A queue can also be connected to multiple senders. It should generally not be
connected to multiple receivers, because they will each get some undefined
subset of the packets, which won't reassemble into the proper heaps. However,
if you set the packet size large enough that every heap is contained in one
packet then it will work.

.. warning::

   Even though the transport is reliable, a stream has a maximum number of
   outstanding heaps. Attempting to send more heaps in parallel than the
   stream is configured to handle can lead to heaps being dropped. This is
   not a problem when using a single thread with
   :py:meth:`spead2.send.InprocStream.send_heap` (because it blocks until the
   heap has been fully added to the queue), but needs to be considered when
   sending heaps in parallel with :py:class:`spead2.send.asyncio.InprocStream`
   or when using multiple threads.

Sending
^^^^^^^

.. py:class:: spead2.InprocQueue()

    .. py:method:: stop()

       Indicate end-of-stream to receivers. It is an error to add any more
       packets after this.

.. py:class:: spead2.send.InprocStream(thread_pool, queue, config)

   :param thread_pool: Thread pool handling the I/O
   :type thread_pool: :py:class:`spead2.ThreadPool`
   :param queue: Queue holding the generated packets
   :type queue: :py:class:`spead2.InprocQueue`
   :param config: Stream configuration
   :type config: :py:class:`spead2.send.StreamConfig`

   .. py:attribute:: queue

      Get the queue passed to the constructor.

.. autoclass:: spead2.send.asyncio.InprocStream(thread_pool, queue, config)

   An asynchronous version of :py:class:`spead2.send.InprocStream`. Refer to
   :ref:`asynchronous-send` for general details about asynchronous transport.

Receiving
^^^^^^^^^

To connect a receiver to the the queue, use
:py:meth:`spead2.recv.Stream.add_inproc_reader`.
