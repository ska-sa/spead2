Debugging
=========

Debug builds
------------
Meson provides standard infrastructure for doing debug builds. Specifically,
these disable optimisation and enable assertions. For C++ builds you can pass
:option:`!--buildtype=debug` when setting up the build directory (note that Meson
supports multiple build directories, so you can keep separate directories for
release and debug builds if you like). For Python builds, you can pass
:option:`!-Csetup-args=--buildtype=debug` to :command:`pip install`. Note that
for an editable install, this option is *sticky*: invoking :command:`pip
install` in future without this option will not reset it to the default,
unless you delete the :file:`build` directory.

Debug logging
-------------
Debug builds do not automatically enable debug-level logging. See the
:doc:`py-logging` documentation for instructions to do that.

Debug symbols for Python wheels
-------------------------------
Occasionally a bug may manifest in a released Python wheel but prove
impossible to reproduce with a locally-compiled version of the package. While
it will not give a great debugging experience (because the code is optimised),
it is possible to install separate debug symbols so that one can get line
numbers from stack traces. Note that this is only supported on Linux.

On the Github page for the release is a file called
:samp:`spead2-{version}-debug.tar.xz`. Unpack it into
:file:`lib/pythonX.X/site-packages/spead2` inside your virtual environment).
You only need to install the :file:`.debug` file matching the Python version
and architecture. It should have the same name as an existing file in the same
directory, but with the :file:`.debug` suffix. Once this is done, GDB should
be able to load the debug symbols from this file. Note that it will only work
for the released wheel from the same version; if you compile a wheel yourself
then the build ID will most likely not match and GDB will not use it.

Reducing worker threads
-----------------------
Numpy creates a lot of worker threads, which can make it more difficult to
find the thread of interest in gdb. Setting the environment variable
:envvar:`OMP_NUM_THREADS` to 1 will reduce the number of threads to sift
through.
