Migrating to version 4
======================

Unlike version 3, version 4 does not make changes to the spead2 API. However, it
replaces the build system and increases the minimum version requirements, which
may have implications for how you install, link to or use spead2.

C++17
-----
The codebase now uses C++17, whereas older versions used C++11. This might
require a newer C++ compiler. See the :doc:`Introduction <introduction>` for
minimum compiler versions.

Additionally, when compiling against the C++ API, you may need to pass
compiler arguments to select at least C++17 (e.g. :option:`!--std=c++17`). GCC
11+ and Clang 16+ support C++17 without a compiler flag, but keep in mind that
your users might use older compilers.

Boost
-----
Boost 1.69+ is now required: from this release, boost_system is
a header-only library. You no longer need to link against any Boost libraries
when linking against spead2.

Meson
-----
The autotools build system has been replaced by `Meson`_. This mainly affects
C++ users, as for Python this is hidden behind the Python packaging
interface. Refer to the :doc:`Introduction <introduction>` for installation
instructions.

The old build system had a number of options to adjust the build. The table
below shows corresponding Meson options:

====================================== =====================================
autotools                              meson
====================================== =====================================
``--enable-debug-symbols``             ``debug=true`` or ``buildtype=...``
``--enable-debug-log``                 ``max_log_level=debug``
``--enable-coverage``                  ``b_coverage=true``
``--disable-optimized``                ``optimization=0`` or ``buildtype=debug``
``--enable-lto``                       ``b_lto=true``
``--enable-shared``                    ``default_library=both``
``--without-program-options``          ``tools=disabled``
``--without-ibv``                      ``ibv=disabled``
``--without-mlx5dv``                   ``mlx5dv=disabled``
``--without-ibv-hw-rate-limit``        ``ibv_hw_rate_limit=disabled``
``--without-pcap``                     ``pcap=disabled``
``--without-cap``                      ``cap=disabled``
``--without-recvmmsg``                 ``recvmmsg=disabled``
``--without-sendmmsg``                 ``sendmmsg=disabled``
``--without-eventfd``                  ``eventfd=disabled``
``--without-posix-semaphores``         ``posix_semaphores=disabled``
``--without-pthread_setaffinity_np``   ``pthread_setaffinity_np=disabled``
``--without-fmv``                      ``fmv=disabled``
``--without-movntdq``                  ``movntdq=disabled``
``--without-cuda``                     ``cuda=disabled``
``--without-gdrapi``                   ``gdrapi=disabled``
====================================== =====================================

Link-time optimization no longer requires intervention to select suitable
versions of :command:`ar` and :command:`ranlib`; Meson takes care of it.

Code generation
---------------
In older versions of spead2, some of the code was generated and included in
the release tarballs. If you used a release, you would be unaware of this, but
trying to build directly from git would require you to run a ``bootstrap.sh``
script.

Meson doesn't have good support for including generated code into releases, so
these generated files are no longer included in the releases, and they are
instead created as part of the build. This requires Python, with the
:mod:`jinja2`, :mod:`pycparser` and :mod:`packaging` packaging installed.

An advantage of this approach is that it is now possible to directly build
from a git checkout without any preparatory steps.

Python configuration
--------------------
When building the Python bindings from source, it was previously only possible
to adjust the build-time configuration by editing source files. With
the new build system, it's now possible to `pass options`_ on the command
line.

.. _pass options: https://meson-python.readthedocs.io/en/latest/how-to-guides/config-settings.html

Python editable installs
------------------------
Meson-python `doesn't support <no-editable_>`_ editable installs with build
isolation. To make an editable install, use ``pip install --no-build-isolation -e .``.

.. _no-editable: https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html
