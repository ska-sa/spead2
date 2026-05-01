Logging
-------
Logging is done with the standard Python :py:mod:`logging` module, and logging
can be configured with the usual utilities. However, in the default build the
debug logging is completely disabled for performance reasons. To enable
it, you need to set the Meson option ``max_log_level=debug``. For example, if
installing with :command:`pip`, use

.. code-block:: sh

   pip install --config-settings=setup-args=-Dmax_log_level=debug .
