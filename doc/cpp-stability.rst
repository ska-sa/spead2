C++ API stability
=================
The C++ API is less stable between versions than the Python API. The
most-derived classes defining specific transports are expected to be stable.
Applications that subclass the base classes to define new transports may be
broken by future API changes, as there is still room for improvement in the
API between these classes and the core.
