A quick introduction to SPEAD
=============================
Before developing a full-fledged application it is a good idea to read all the
details of the :download:`SPEAD <../SPEAD_Protocol_Rev1_2012.pdf>` protocol, but
for the purposes of the tutorial we'll provide a brief overview here.

SPEAD is a message-based protocol, where the messages are called :dfn:`heaps`.
A sequence of heaps all sent to the same receiver is called a :dfn:`stream`.
Heaps may be very large, so to facilitate transmission over protocols (such as
UDP) that have a limited message size, heaps may be fragmented into multiple
:dfn:`packets`. The receiver collects all the packets that belong to the same
heap and reassembles them.

Each heap contains a number of :dfn:`items`. Each item has

- a :dfn:`name`, which is a short string that can be used to look up the item.
  Typically it's a valid programming language identifier, but this is not
  required;
- an :dfn:`ID`, which is a small integer used to identify the item in the
  protocol. You can either assign your own IDs or you can let spead2 assign
  incremental IDs for you.
- a :dfn:`description`, which is a longer string meant to document the item
  for humans;
- a :dfn:`shape`, which indicates the dimensions of a multi-dimensional array.
  This can be empty for scalar values;
- a :dfn:`type`; and
- a :dfn:`value`.

It is quite common for a heap to contain only the ID and value of an item, to
avoid repeating all the other information if it does not change. The other
information is packaged into a special object called a :dfn:`descriptor` and
included into a heap. At a minimum, descriptors are sent in the first heap of
the stream, but may also be sent periodically (for the benefit of receivers
who missed the initial descriptors).

The Python bindings in spead2 contain utilities to manage all this. In
particular, an :dfn:`item group` holds all the items that are going to be
transmitted on a stream. It tracks which descriptors and values have already
been transmitted.  The general process is thus to modify some of the items,
request a heap from the item group, and transmit it on a stream.

The C++ bindings operate at a lower level. You will need to explicitly
construct each heap from the items you want, including any descriptors.
