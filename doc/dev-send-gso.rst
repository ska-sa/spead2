Generic segmentation offload
============================
Linux supports a mechanism called :dfn:`generic segmentation offload` (GSO) to
reduce packet overheads when transmitting UDP data through the kernel
networking stack. A good overview can be found on `Cloudflare's blog`_, but
the basic idea is this:

.. _Cloudflare's blog: https://blog.cloudflare.com/accelerating-udp-packet-transmission-for-quic/

1. Userspace concatenates multiple smaller packets into one mega-packet for
   submission to the kernel.
2. Most of the networking stack operates on the mega-packet.
3. As late as possible (and possibly on the NIC) the mega-packet is
   re-segmented into the original packets.

The re-segmentation uses a user-supplied parameter (socket option) indicating
the size of the original packets. This imposes a limitation that the original
packets were all the same size, except perhaps for the last one in the
mega-packet.

The support for this in spead2 is dependent on the :manpage:`sendmmsg(2)`
support. While there is no fundamental reason GSO can't be used without
:manpage:`sendmmsg(2)`, supporting it would complicate the code significantly,
and GSO is a much more recent feature so it is unlikely that this combination
would ever be needed.

Run-time detection of support is unfortunately rather complicated. The simple
part is that an older kernel will not support the socket option. If that
occurs, we simply disable GSO for the stream. A more tricky problem is that
actually sending the message may fail for several reasons:

- Fragmentation doesn't seem to be supported, so if the segment size is bigger
  than the MTU, it will fail.
- If hardware checksumming is disabled (or presumably if it is not supported),
  it will fail.

To cope with this complication, a state machine is used. It has four possible
states:

- **active**: the socket option is set to a positive value
- **inactive**: the socket option is set to zero, but we may still transition
  to active
- **probe**: the last send in active state failed; the socket option is now
  set to zero and we're retrying
- **disabled**: the socket option is set to zero, and we will never try to set
  it again.

If send fails while in state **active**, we switch to state **probe** and try
again (without GSO). If that succeeds, we conclude that GSO is non-functional
for this stream and permanently go to **disabled**. If that also fails, we
conclude that the problem was unrelated to GSO and return to **inactive**.
