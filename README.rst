spead2
======

.. image:: https://readthedocs.org/projects/spead2/badge/?version=latest
   :target: https://readthedocs.org/projects/spead2/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/ska-sa/spead2/actions/workflows/test.yml/badge.svg
   :target: https://github.com/ska-sa/spead2/actions/workflows/test.yml
   :alt: Test Status

.. image:: https://coveralls.io/repos/github/ska-sa/spead2/badge.svg
   :target: https://coveralls.io/github/ska-sa/spead2
   :alt: Coverage Status

spead2 is an implementation of the SPEAD_ protocol, with both Python and C++
bindings. The *2* in the name indicates that this is a new implementation of
the protocol; the protocol remains essentially the same. Compared to the
PySPEAD_ implementation, spead2:

- is at least an order of magnitude faster when dealing with large heaps;
- correctly implements several aspects of the protocol that were implemented
  incorrectly in PySPEAD (bug-compatibility is also available);
- correctly implements many corner cases on which PySPEAD would simply fail;
- cleanly supports several SPEAD flavours (e.g. 64-40 and 64-48) in one
  module, with the receiver adapting to the flavour used by the sender;
- supports Python 3;
- supports asynchronous operation, using asyncio_.

For more information, refer to the documentation on readthedocs_.

.. _SPEAD: https://casper.berkeley.edu/wiki/SPEAD
.. _PySPEAD: https://github.com/ska-sa/PySPEAD/
.. _asyncio: https://docs.python.org/3/library/asyncio.html
.. _readthedocs: http://spead2.readthedocs.io/en/latest/
