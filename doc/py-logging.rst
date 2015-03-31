Logging
-------
Logging is done with the standard Python :py:mod:`logging` module, and logging
can be configured with the usual utilities. However, in the default build the
debug logging is completely disabled for performance reasons [#]_.

.. [#] Logging is done from separate C threads, which have to wait for
  Python's Global Interpreter Lock (GIL) in order to do logging.
