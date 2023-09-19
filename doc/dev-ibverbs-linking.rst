Linking to ibverbs
==================

The Python wheels for spead2 support :doc:`ibverbs <py-ibverbs>`. But what
happens if the ibverbs library isn't installed on the user's machine? While
for pcap we simply bundle a copy of the library into the wheel, this is
problematic for ibverbs: it uses configuration files in
:file:`/etc/libibverbs.d` to configure drivers, which are themselves contained
in other shared libraries. At best, one would end up with a mix of code from
the wheel and from the operating system, which could easily lead to
compatibility problems.

Instead, the ibverbs libraries (:file:`libibverbs.so`, :file:`librdmacm.so`
and :file:`libmlx5.so`) are loaded dynamically at runtime with `dlopen(3)`_.
If the library is not found, the corresponding functionality in spead2 is
disabled.

Unfortunately, dynamic loading is not easy to use: `dlsym(3)`_ returns a raw
pointer with no type information. One needs to check for errors and cast the
pointer to the proper type before it can be invoked. This leads to a lot of
boilerplate code, which in spead2 is generated at build time by
:file:`gen/gen_loader.py` for each of the libraries.

The process starts with the signatures for the functions that are to be
written, which are parsed using pycparser_. The actual header files for these
functions are not suitable for pycparser (it only handles standard C99), so
the script has embedded declarations for all the functions we wish to wrap. It
then emits a header file and a source file, both generated from templates
using jinja2_. It generates a private initialisation function that opens the
library (if possible) and extracts the symbols from it. It also generates the
following for each function:

- A declaration of the function pointer. It has the same name as the original
  function, but is in the ``spead2`` namespace to prevent symbol conflicts.
- An implementation that calls the initialisation function (using
  :cpp:func:`std::call_once`, to avoid initialising multiple times), then
  calls through the function pointer. The function pointer is defined to
  initially point to this implementation, so that the first use of the
  function pointer does the initialisation.
- A "stub" implementation. If there was a problem loading the library, the
  function pointers are all changed to point to the stub implementations,
  which when called will re-throw the exception that occurred during
  initialisation.

Additionally, some functions are optional: they were added in later versions
of the wrapped library, and spead2 has fallback code that can be used if the
function is not available. For those, the generator additionally creates:

- A "missing" implementation, which throws a :cpp:class:`std::system_error`
  with error code :c:macro:`EOPNOTSUPP`. If the function is not found during
  initialisation, the function pointer is set to point at this implementation.
- A (public) function which has the same name but prefixed with ``has_``,
  which returns a boolean to indicate whether the function is present.

Changing a function pointer during initialisation ensures that only the first
call incurs any overheads from the wrapping; after that, calls are made
directly to the target function.

.. _dlopen(3): https://man7.org/linux/man-pages/man3/dlopen.3.html
.. _dlsym(3): https://man7.org/linux/man-pages/man3/dlsym.3.html
.. _pycparser: https://github.com/eliben/pycparser
.. _jinja2: https://jinja.palletsprojects.com/
