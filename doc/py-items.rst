Items and item groups
---------------------
Each data item that can be communicated over SPEAD is described by a
:py:class:`spead2.Descriptor`. Items combine a descriptor with a current
value, and a version number that is used to detect which items have been
changed (either in the library when transmitting, or by the user when
receiving).

.. py:currentmodule:: spead2

.. autoclass:: spead2.Descriptor

   .. autoattribute:: itemsize_bits
   .. automethod:: is_variable_size
   .. automethod:: dynamic_shape
   .. automethod:: compatible_shape

.. autoclass:: spead2.Item(\*args, \*\*kwargs, value=None)

   .. autoattribute:: value
   .. autoinstanceattribute:: version
      :annotation:

.. autoclass:: spead2.ItemGroup

   .. automethod:: add_item
   .. automethod:: keys
   .. automethod:: values
   .. automethod:: items
   .. automethod:: ids
   .. automethod:: update
