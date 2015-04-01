Logging
=======
By default, log messages are all written to standard error. However, the
logging function can be replaced by calling
:cpp:func:`spead2::set_log_function`.

.. cpp:function:: void spead2::set_log_function(std::function<void(log_level, const std::string &)>)
