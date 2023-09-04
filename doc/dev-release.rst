Release checklist
=================

- Update the version number and changelog in :file:`doc/changelog.rst`
- Update the version number in :file:`VERSION.txt`
- Update the shared library version number in :file:`meson.build`:

  - If there are ABI changes, update the first number and reset the second to zero.
  - Otherwise, increment the second number.

- Check that :file:`.pyi` stubs have been updated
- Check that Github Actions successfully tested the release and built wheels
- Install the sdist from Github Actions and check that it passes pytest
- Install a wheel from Github Actions and check that it passes pytest
- Tag the release
- Run :command:`git push --tags`
- Upload the sdist and wheels to PyPI with twine_
- Upload the sdist and debug symbols to Github release
- Check that readthedocs_ has updated itself

.. _twine: https://twine.readthedocs.io/
.. _readthedocs: https://spead2.readthedocs.io/
