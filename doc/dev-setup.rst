Getting started with development
================================

Python setup
------------
Refer to the :doc:`introduction` for the prerequisite packages (particularly
:ref:`py-install-source`). You will also need ninja-build_, and a
Python virtual environment. You can use any tool you like to create the
virtual environment, but it must be located outside of the working copy of the
spead2 repository [#meson-dir-bug]_. I like pyenv_ (with pyenv-virtualenv_), but you can also
use venv_ or virtualenv_ directly. It's also a good idea to install ccache_ or
sccache_, as it will make recompilation much faster.

Check out a copy of spead2 from git, and make it your current directory.
Install pre-commit_ (e.g., with :command:`pip install pre-commit`) and run
:command:`pre-commit install` to set it up. This will ensure that commits pass
a set of static analysis checks.

Inside your virtual environment, install the build dependencies. You can find
a list of them right at the top of :file:`pyproject.toml`.

You're now ready to make an "editable" installation. This is an installation
that will use the files inside your working copy (recompiling C++ code if
necessary), so that you don't need to explicitly install after each change.
To do so, run

.. code-block:: sh

   pip install --no-build-isolation -e .

See the meson-python_ documentation for more information about the limitations
of editable installs. Also install the development runtime dependencies:

.. code-block:: sh

   pip install -r requirements.txt -r requirements-3.12.txt

Now you should be able to run the unit tests by executing :command:`pytest`.
It is expected that some tests will be skipped, because they require specific
hardware. Running :command:`pytest -ra` will show the reasons for skipped
tests. You should expect to see something like::

    SKIPPED [23] tests/test_passthrough.py:577: Envar SPEAD2_TEST_IBV_INTERFACE_ADDRESS not set

If you're running the latest version of Python, it's possible that
:mod:`numba` does not yet support it and so was not installed above. In that
case you will have additional skipped tests e.g.::

    SKIPPED [1] tests/test_recv_chunk_stream.py:30: could not import 'numba': No module named 'numba'
    SKIPPED [1] tests/test_recv_chunk_stream_group.py:30: could not import 'numba': No module named 'numba'

MacOS will have some additional skipped tests. On Linux, there should be no
other skipped tests if you have all the optional dependencies installed. If
there are other tests skipped, it is not a show-stopper; it just means you'll
need to rely on the CI to run those tests for you.

C++ setup
---------
You should start by following the steps for Python. Most of the functionality
is only tested via the Python unit tests, so you will need to be able to run
those even if you are only interested in working on the C++ bindings.

You can then follow the :ref:`cxx-install` instructions to build the C++
bindings (you can skip :command:`meson install`). From the build directory,
also run :command:`meson test` to run the C++ unit tests. These are a small
set of tests that cover functionality that is not practical to test from the
Python API.

Documentation
-------------
To install the necessary Python requirements, run :command:`pip install -r
requirements-readthedocs.txt`. You will also need doxygen_ and :program:`make`. Then
change to the :file:`doc` directory and run :command:`make`. This will build
documentation in :file:`doc/_build/html`. It is unfortunately normal for there
to be a large number of warnings about duplicates.

Coding style
------------
The first rule is just to adhere the existing style. Python code uses black_
and ruff_ to enforce style, so if you deviate from the style those tools will
guide you back on track. The Python code generally does not use inline type
annotations, because annotations in the :file:`.pyi` files take precedence
(and spead2 pre-dates Python 3 annotation syntax). New code (particularly in
tests) can be annotated, but it is not required.

Identifiers use US English spelling, but comments, log messages and
documentation favour UK spelling.

The C++ code is less consistent in style, but here are some guidelines:

- Use 4 spaces for indentation (**never** tabs).
- Opening braces go on their own line (Allman style). An exception is that a
  function may be written entirely on one line if it is very short.
- Do not use trailing commas.
- Do not add a level of indentation inside namespaces.
- When two levels of namespaces start and end at the same point, use the
  C++17 nested namespace syntax:

  .. code-block:: c++

     namespace spead2::recv
     {
     /* Stuff */
     } // namespace spead2::recv

- When closing a namespace or a ``#endif``, use a comment to indicate what is
  being closed, unless it is visually obvious (nearby and without further
  nesting).
- Be sparing with using ``auto`` to declare local variables. It should ideally
  be possible for the user to guess what the type is just by inspecting the
  code. Good reasons to use ``auto`` include:

  - The type is impossible to specify safely, because it is a lambda, or an
    implementation-defined type that could change in future.
  - It is an integer type, and explicitly naming the type could inadvertently
    cause type conversions if the type of the expression later changed.
  - The type is obvious from the initialiser, such as

    .. code-block:: c++

       auto foo = std::make_unique<Foo>(1);

  - The type is exceedingly long to write out (iterator types are a good
    example).

- Start a class with friends, followed by typedefs, member variables, and
  finally member functions. Put private members before public ones, unless a
  specific order is required (for example, to optimise memory layout or to
  control initialisation/destruction order).
- Line comments (``//``) should only be used for one-line comments (maybe two
  at a push). Use block comments (``/* */``) for longer blocks of text.
- If a member function has an empty body and exists only to implement a
  concept, it can use anonymous parameters if they are self-explanatory.
  Otherwise, unused parameters should be named but have the
  ``[[maybe_unused]]`` attribute. In some cases a particular compiler may
  still generate warnings after applying the attribute (GCC 9 has been seen to
  do this); in such cases one should place the parameter name inside
  ``/* */``.

Committing
----------
Before committing, remember to run :command:`pre-commit install` to set up
pre-commit. One of the pre-commit hooks checks that the requirements files are
up to date, and (at the time of writing) depends on having both
:command:`python3.8` and :command:`python3.12` commands on the path. If you're
not touching the requirements, you can skip this hook by setting the
environment variable :envvar:`SKIP=pip-compile` when committing.

.. _ninja-build: https://ninja-build.org/
.. _pyenv: https://github.com/pyenv/pyenv/
.. _pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv
.. _venv: https://docs.python.org/3/library/venv.html
.. _virtualenv: https://virtualenv.pypa.io/en/latest/user_guide.html
.. _ccache: https://ccache.dev/
.. _sccache: https://github.com/mozilla/sccache
.. _pre-commit: https://pre-commit.com/
.. _black: https://black.readthedocs.io/
.. _ruff: https://beta.ruff.rs/docs/
.. _meson-python: https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html
.. _doxygen: https://www.doxygen.nl/

.. [#meson-dir-bug] Meson will show a long error starting with
   "ERROR: Tried to form an absolute path to a dir in the source tree."
   There is also a Meson `bug
   <https://github.com/mesonbuild/meson/issues/12217>`_ that causes this error
   to appear if the source directory is a prefix *as a string* of the virtual
   environment path, even if the virtual environment is not inside the source
   directory.

Making a pull request
---------------------
spead2 uses the normal Github workflow for pull requests. There are many
guides on the internet to writing good pull requests, such as
`this one <perfect-pr_>`_ or `this one <unwritten-pr_>`_.
A few points to note for spead2:

- Don't add to the changelog. The changelog for each release is generally
  prepared just prior to each release. However, it is a good idea to write a
  meaningful title for the pull request that could become the changelog entry.
- Once a pull request has been reviewed, don't force-push changes. Doing so
  prevents the reviewer from seeing the difference between the
  previously-reviewed version and your update. If you're a stickler for a neat
  commit history, ask if you can rebase just prior to merging.

.. _perfect-pr: https://github.blog/2015-01-21-how-to-write-the-perfect-pull-request/
.. _unwritten-pr: https://www.atlassian.com/blog/git/written-unwritten-guide-pull-requests
