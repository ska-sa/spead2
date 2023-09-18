Send rate limiting
==================
The basic principle behind the rate limiting is that is that after sending a
packet, one should sleep for some time before sending the next packet, to
create gaps on the wire. However, there are a number of challenges:

1. Checking the time has some cost, and sleeping has quite a large cost.
   Sleeping after every packet can add so much cost that one can't keep up with
   the desired rate.
2. The OS can be late to wake up the process after a sleep. If not compensated
   for, this oversleeping time will reduce the achieved rate.
3. Naïvely catching up from oversleeping by transmitting as fast as possible
   can lead to a large burst of back-to-back packets that overwhelm the
   receiver.

The first point is addressed by sending several packets at a time without
sleeping in between. Apart from reducing the number of sleeps, this also
allows multiple packets to be batched together for transmission with APIs such
as :cpp:func:`sendmmsg`. This is the `burst_size` parameter in
:py:class:`~spead2.send.StreamConfig`.

The remaining points are handled by using two rates: the "standard" rate that
the user requested, and a "catch-up" rate that is used when it is necessary to
catch up after oversleeping, and which is specified indirectly via the
`burst_rate_ratio` parameter in :py:class:`~spead2.send.StreamConfig`.

.. note::

   While both parameters have "burst" in the name, they control two different
   bursting mechanisms: sending small amounts with no sleeping at all, and
   sending larger amounts at the burst rate to catch up on oversleeping.

The two rates are managed by keeping two lower bounds for sending the next
burst. For the standard rate, the time is incremented after each burst
according only to the size of the burst, without considering actual
transmission times. For the burst rate, the time for the next burst is the
time that the current burst was *actually* sent plus the size over the rate.

The above all assumes that the producer always has some data to send, but in
some applications the sender may go dormant for some extended time. When it
starts again, a naïve implementation might interpret this dormant period as
oversleeping and switch to the burst rate to catch up. To avoid this, the rate
mechanism handles this case specially by adjusting the standard rate lower
bound such that no catching up is required.

State machine
-------------
The :cpp:class:`~spead2::send::writer` is a state machine which the following
states:

- **New**: freshly constructed. Nothing happens in this state, because the
  associated stream has not yet been set.
- **Active**: The writer is either executing code or has made internal
  arrangements to be woken up (for example, it has asynchronously sent some
  packets and is waiting for the completion handler).
- **Sleeping**: The rate limiter is sleeping.
- **Empty**: All the queued heaps have been sent, and we are waiting for the
  user to provide more.

.. tikz:: State transitions
   :libs: positioning

   \tikzset{
       state/.style={minimum width=2cm, minimum height=1cm, draw, rounded corners},
       every to/.style={font={\small},append after command={[draw,>=latex,->]}},
       loop/.style={to path={.. controls +(80:-1) and +(100:-1) .. (\tikztotarget) \tikztonodes}},
   }
   \node[state] (active) {Active};
   \node[state,above=of active] (new) {New};
   \node[state,left=2cm of active] (sleeping) {Sleeping};
   \node[state,right=2cm of active] (empty) {Empty};
   \draw (new) to [edge label={\ttfamily start()}] (active);
   \draw (active) to[loop, edge label'={\ttfamily SUCCESS}] (active);
   \draw (active) to[bend left=10, edge label={\ttfamily EMPTY}] (empty);
   \draw (active) to[bend right=10, edge label'={\ttfamily SLEEP}] (sleeping);
   \draw (empty) to[bend left=10, edge label={heap added}] (active);
   \draw (sleeping) to[bend right=10, edge label'={timer expires}] (active);

The transitions from **Active** are labelled by the return value from
:cpp:func:`spead2::send::writer::get_packet`. Transitions back to **Active**
are achieved by calling :cpp:func:`spead2::send::writer::wakeup`.

Time precision
--------------
Even though sleeping is not very precise, it has turned out to be necessary to
do time arithmetic with very high (sub-nanosecond) precision. The reason is
that standard rate lower bound will typically be incremented by the same
amount for each burst, and hence any rounding error will be in the same
direction each time. As an example, suppose the desired rate is 40 Gb/s, and
each burst is 65536 bytes. Then the time between bursts should be
13107.2 ns. If arithmetic were done at nanosecond precision, that would round to
13107 ns each time, giving an actual rate of 40.0006 Gb/s. The higher the rate
or the smaller the burst, the greater the relative error.

This is handled by representing absolute times as the sum of two parts: a
:cpp:class:`!time_point` of the timer class (typically nanosecond resolution),
and an additional correction in double precision (always between 0 and 1 units
of :cpp:class:`!time_point`). When actually sleeping, only the first
("coarse") part is used, since that is all the precision that can be given to
the timer. The correction term accumulates the rounding errors so that they do
not get lost. Keeping the correction in the interval [0, 1) simplifies
comparison of precise times.
