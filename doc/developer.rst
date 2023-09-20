Developer documentation
=======================

Development processes
---------------------

.. toctree::
   :maxdepth: 1

   dev-setup
   dev-layout
   dev-debug
   dev-release

Design
------

This section documents internal design decisions that users will generally not
need to be aware of, although some of it may be useful if you plan to subclass
the C++ classes to extend functionality.

.. toctree::
   :maxdepth: 1

   dev-py-binding
   dev-recv-locking
   dev-recv-destruction
   dev-recv-chunk-group
   dev-send-rate-limit
   dev-ibverbs-linking
