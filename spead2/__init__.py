# Copyright 2015 SKA South Africa
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numbers as _numbers
import logging

import six
import numpy as _np

import spead2._spead2
from spead2._spead2 import (             # noqa: F401
    Flavour, ThreadPool, Stopped, Empty,
    MemoryAllocator, MmapAllocator, MemoryPool, InprocQueue,
    BUG_COMPAT_DESCRIPTOR_WIDTHS,
    BUG_COMPAT_SHAPE_BIT_1,
    BUG_COMPAT_SWAP_ENDIAN,
    BUG_COMPAT_PYSPEAD_0_5_2,
    NULL_ID,
    HEAP_CNT_ID,
    HEAP_LENGTH_ID,
    PAYLOAD_OFFSET_ID,
    PAYLOAD_LENGTH_ID,
    DESCRIPTOR_ID,
    STREAM_CTRL_ID,
    DESCRIPTOR_NAME_ID,
    DESCRIPTOR_DESCRIPTION_ID,
    DESCRIPTOR_SHAPE_ID,
    DESCRIPTOR_FORMAT_ID,
    DESCRIPTOR_ID_ID,
    DESCRIPTOR_DTYPE_ID,
    CTRL_STREAM_START,
    CTRL_DESCRIPTOR_REISSUE,
    CTRL_STREAM_STOP,
    CTRL_DESCRIPTOR_UPDATE,
    MEMCPY_STD,
    MEMCPY_NONTEMPORAL)
try:
    from spead2._spead2 import IbvContext      # noqa: F401
except ImportError:
    pass
from spead2._version import __version__       # noqa: F401


_logger = logging.getLogger(__name__)
_UNRESERVED_ID = 0x1000      #: First ID that can be auto-allocated

_FASTPATH_NONE = 0
_FASTPATH_IMMEDIATE = 1
_FASTPATH_NUMPY = 2


if six.PY2:
    def _bytes_to_str_ascii(b):
        b.decode('ascii')  # Just to check validity, throw away unicode object
        return b
else:
    # Python 3
    def _bytes_to_str_ascii(b):
        return b.decode('ascii')


def _shape_elements(shape):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def parse_range_list(ranges):
    """Split a string like 2,3-5,8,9-11 into a list of integers. This is
    intended to ease adding command-line options for dealing with affinity.
    """
    if not ranges:
        return []
    parts = ranges.split(',')
    out = []
    for part in parts:
        fields = part.split('-', 1)
        if len(fields) == 2:
            start = int(fields[0])
            end = int(fields[1])
            out.extend(range(start, end + 1))
        else:
            out.append(int(fields[0]))
    return out


class Descriptor(object):
    """Metadata for a SPEAD item.

    There are a number of restrictions on the parameters, which will cause
    `ValueError` to be raised if violated:

    - At most one element of `shape` can be `None`.
    - Exactly one of `dtype` and `format` must be non-`None`.
    - If `dtype` is specified, `shape` cannot have any unknown dimensions.
    - If `format` is specified, `order` must be 'C'
    - If `dtype` is specified, it cannot contain objects, and must have
      positive size.

    Parameters
    ----------
    id : int
        SPEAD item ID
    name : str
        Short item name, suitable for use as a key
    description : str
        Long item description
    shape : sequence
        Dimensions, with `None` indicating a variable-size dimension
    dtype : numpy data type, optional
        Data type, or `None` if `format` will be used instead
    order : {'C', 'F'}
        Indicates C-order or Fortran-order storage
    format : list of pairs, optional
        Structure fields for generic (non-numpy) type. Each element of the list
        is a tuple of field code and bit length.
    """

    def __init__(self, id, name, description, shape, dtype=None, order='C', format=None):
        shape = tuple(shape)
        unknowns = sum([x is None for x in shape])
        if unknowns > 1:
            raise ValueError('Cannot have multiple unknown dimensions')
        if dtype is not None:
            dtype = _np.dtype(dtype)
            if dtype.hasobject:
                raise ValueError('Cannot use dtype that has reference-counted objects')
            if format is not None:
                raise ValueError('Only one of dtype and format can be specified')
            if dtype.itemsize == 0:
                raise ValueError('Cannot use zero-sized dtype')
            if unknowns > 0:
                raise ValueError('Cannot have unknown dimensions when using numpy descriptor')
            self._internal_dtype = dtype
        else:
            if format is None:
                raise ValueError('One of dtype and format must be specified')
            if order != 'C':
                raise ValueError("When specifying format, order must be 'C'")
            self._internal_dtype = self._parse_format(format)

        if order not in ['C', 'F']:
            raise ValueError("Order must be 'C' or 'F'")
        self.id = id
        self.name = name
        self.description = description
        self.shape = shape
        self.dtype = dtype
        self.order = order
        self.format = format
        if not self._internal_dtype.hasobject:
            self._fastpath = _FASTPATH_NUMPY
        elif (not shape and
                dtype is None and
                len(format) == 1 and
                format[0][0] in ('u', 'i') and
                self._internal_dtype.hasobject):
            self._fastpath = _FASTPATH_IMMEDIATE
        else:
            self._fastpath = _FASTPATH_NONE

    @classmethod
    def _parse_numpy_header(cls, header):
        try:
            d = _np.lib.utils.safe_eval(header)
        except SyntaxError as e:
            msg = "Cannot parse descriptor: %r\nException: %r"
            raise ValueError(msg % (header, e))
        if not isinstance(d, dict):
            msg = "Descriptor is not a dictionary: %r"
            raise ValueError(msg % d)
        keys = list(d.keys())
        keys.sort()
        if keys != ['descr', 'fortran_order', 'shape']:
            msg = "Descriptor does not contain the correct keys: %r"
            raise ValueError(msg % (keys,))
        # Sanity-check the values.
        if (not isinstance(d['shape'], tuple) or
                not all([isinstance(x, _numbers.Integral) and x >= 0 for x in d['shape']])):
            msg = "shape is not valid: %r"
            raise ValueError(msg % (d['shape'],))
        if not isinstance(d['fortran_order'], bool):
            msg = "fortran_order is not a valid bool: %r"
            raise ValueError(msg % (d['fortran_order'],))
        try:
            dtype = _np.dtype(d['descr'])
        except TypeError:
            msg = "descr is not a valid dtype descriptor: %r"
            raise ValueError(msg % (d['descr'],))
        order = 'F' if d['fortran_order'] else 'C'
        return d['shape'], order, dtype

    @classmethod
    def _make_numpy_header(self, shape, dtype, order):
        return "{{'descr': {!r}, 'fortran_order': {!r}, 'shape': {!r}}}".format(
            _np.lib.format.dtype_to_descr(dtype), order == 'F',
            tuple(shape))

    @classmethod
    def _parse_format(cls, fmt):
        """Attempt to convert a SPEAD format specification to a numpy dtype.
        Where necessary, `O` is used.

        Raises
        ------
        ValueError
            If the format is illegal
        """
        fields = []
        if not fmt:
            raise ValueError('empty format')
        for code, length in fmt:
            if length <= 0:
                if length == 0:
                    raise ValueError('zero-length field (bug_compat mismatch?)')
                else:
                    raise ValueError('negative-length field')
            if ((code in ('u', 'i') and length in (8, 16, 32, 64)) or
                    (code == 'f' and length in (32, 64))):
                fields.append('>' + code + str(length // 8))
            elif code == 'b' and length == 8:
                fields.append('?')
            elif code == 'c' and length == 8:
                fields.append('S1')
            else:
                if code not in ['u', 'i', 'b']:
                    raise ValueError('illegal format ({}, {})'.format(code, length))
                fields.append('O')
        return _np.dtype(','.join(fields))

    @property
    def itemsize_bits(self):
        """Number of bits per element"""
        if self.dtype is not None:
            return self.dtype.itemsize * 8
        else:
            return sum(x[1] for x in self.format)

    def is_variable_size(self):
        """Determine whether any element of the size is dynamic"""
        return any([x is None for x in self.shape])

    def allow_immediate(self):
        """Called by the C++ interface to determine whether sufficiently small
        items should be encoded as immediates.

        Variable-size objects cannot be immediates because there is no way to
        determine the true payload size. Types with a non-integral number of
        bytes are banned because the protocol does not specify where the
        padding should go, and PySPEAD's encoder and decoder disagree, so it
        is best not to send them at all.
        """
        return not self.is_variable_size() and (
            self.dtype is not None or self.itemsize_bits % 8 == 0)

    def dynamic_shape(self, max_elements):
        """Determine the dynamic shape, given incoming data that is big enough
        to hold `max_elements` elements.
        """
        known = 1
        unknown_pos = -1
        for i, x in enumerate(self.shape):
            if x is not None:
                known *= x
            else:
                assert unknown_pos == -1, 'Shape has multiple unknown dimensions'
                unknown_pos = i
        if unknown_pos == -1:
            return self.shape
        else:
            shape = list(self.shape)
            if known == 0:
                shape[unknown_pos] = 0
            else:
                shape[unknown_pos] = max_elements // known
            return shape

    def compatible_shape(self, shape):
        """Determine whether `shape` is compatible with the (possibly
        variable-sized) shape for this descriptor"""
        if len(shape) != len(self.shape):
            return False
        for x, y in zip(self.shape, shape):
            if x is not None and x != y:
                return False
        return True

    @classmethod
    def from_raw(cls, raw_descriptor, flavour):
        dtype = None
        format = None
        if raw_descriptor.numpy_header:
            header = _bytes_to_str_ascii(raw_descriptor.numpy_header)
            shape, order, dtype = cls._parse_numpy_header(header)
            if flavour.bug_compat & BUG_COMPAT_SWAP_ENDIAN:
                dtype = dtype.newbyteorder()
        else:
            shape = raw_descriptor.shape
            order = 'C'
            format = raw_descriptor.format
        return cls(
            raw_descriptor.id,
            _bytes_to_str_ascii(raw_descriptor.name),
            _bytes_to_str_ascii(raw_descriptor.description),
            shape, dtype, order, format)

    def to_raw(self, flavour):
        raw = spead2._spead2.RawDescriptor()
        raw.id = self.id
        raw.name = self.name.encode('ascii')
        raw.description = self.description.encode('ascii')
        raw.shape = self.shape
        if self.dtype is not None:
            if flavour.bug_compat & BUG_COMPAT_SWAP_ENDIAN:
                dtype = self.dtype.newbyteorder()
            else:
                dtype = self.dtype
            raw.numpy_header = self._make_numpy_header(
                self.shape, dtype, self.order).encode('ascii')
        else:
            raw.format = self.format
        return raw


class Item(Descriptor):
    """A SPEAD item with a value and a version number.

    Parameters
    ----------
    value : object, optional
        Initial value
    """

    def __init__(self, *args, **kw):
        value = kw.pop('value', None)
        super(Item, self).__init__(*args, **kw)
        self._value = value
        self.version = 1   #: Version number

    @property
    def value(self):
        """Current value. Assigning to this will increment the version number.
        Assigning `None` will raise `ValueError` because there is no way to
        encode this using SPEAD.

        .. warning:: If you modify a mutable value in-place, the change will
          not be detected, and the new value will not be transmitted. In this
          case, either manually increment the version number, or reassign the
          value.
        """
        return self._value

    @value.setter
    def value(self, new_value):
        if new_value is None:
            raise ValueError("Item value cannot be set to None")
        self._value = new_value
        self.version += 1

    @classmethod
    def _read_bits(cls, raw_value):
        """Generator that takes a memory view and provides bitfields from it.
        After creating the generator, call `send(None)` to initialise it, and
        thereafter call `send(need_bits)` to obtain that many bits.
        """
        have_bits = 0
        bits = 0
        byte_source = iter(raw_value)
        result = 0
        while True:
            need_bits = yield result
            while have_bits < need_bits:
                try:
                    bits = (bits << 8) | int(next(byte_source))
                    have_bits += 8
                except StopIteration:
                    return
            result = int(bits >> (have_bits - need_bits))
            bits &= (1 << (have_bits - need_bits)) - 1
            have_bits -= need_bits

    @classmethod
    def _write_bits(cls, array):
        """Generator that fills a `bytearray` with provided bits. After
        creating the generator, call `send(None)` to initialise it, and
        thereafter call `send((value, bits))` to add that many bits into
        the array. You must call `close()` to flush any partial bytes."""
        pos = 0
        current = 0    # bits not yet written into array
        current_bits = 0
        try:
            while True:
                (value, bits) = yield
                if value < 0 or value >= (1 << bits):
                    raise ValueError('Value is out of range for number of bits')
                current = (current << bits) | value
                current_bits += bits
                while current_bits >= 8:
                    array[pos] = current >> (current_bits - 8)
                    current &= (1 << (current_bits - 8)) - 1
                    current_bits -= 8
                    pos += 1
        except GeneratorExit:
            if current_bits > 0:
                current <<= (8 - current_bits)
                array[pos] = current

    def _load_recursive(self, shape, gen):
        """Recursively create a multidimensional array (as lists of lists)
        from a bit generator.
        """
        if len(shape) > 0:
            ans = []
            for i in range(shape[0]):
                ans.append(self._load_recursive(shape[1:], gen))
        else:
            fields = []
            for code, length in self.format:
                field = None
                raw = gen.send(length)
                if code == 'u':
                    field = raw
                elif code == 'i':
                    field = raw
                    # Interpret as 2's complement
                    if field >= 1 << (length - 1):
                        field -= 1 << length
                elif code == 'b':
                    field = bool(raw)
                elif code == 'c':
                    field = six.int2byte(raw)
                elif code == 'f':
                    if length == 32:
                        field = _np.uint32(raw).view(_np.float32)
                    elif length == 64:
                        field = _np.uint64(raw).view(_np.float64)
                    else:
                        raise ValueError('unhandled float length {0}'.format((code, length)))
                else:
                    raise ValueError('unhandled format {0}'.format((code, length)))
                fields.append(field)
            if len(fields) == 1:
                ans = fields[0]
            else:
                ans = tuple(fields)
        return ans

    def _store_recursive(self, dims, value, gen):
        if dims > 0:
            for sub in value:
                self._store_recursive(dims - 1, sub, gen)
        else:
            if len(self.format) == 1:
                value = (value,)
            for (code, length), field in zip(self.format, value):
                raw = None
                if code == 'u':
                    raw = int(field)
                    if raw < 0 or raw >= (1 << length):
                        raise ValueError('{} is out of range for u{}'.format(raw, length))
                elif code == 'i':
                    top_bit = 1 << (length - 1)
                    raw = int(field)
                    if raw < -top_bit or raw >= top_bit:
                        raise ValueError('{} is out of range for i{}'.format(field, length))
                    # convert to 2's complement
                    if raw < 0:
                        raw += 2 * top_bit
                elif code == 'b':
                    raw = 1 if field else 0
                elif code == 'c':
                    raw = ord(field)
                elif code == 'f':
                    if length == 32:
                        raw = _np.float32(field).view(_np.uint32)
                    elif length == 64:
                        raw = _np.float64(field).view(_np.uint64)
                    else:
                        raise ValueError('unhandled float length {0}'.format((code, length)))
                else:
                    raise ValueError('unhandled format {0}'.format((code, length)))
                gen.send((raw, length))

    def set_from_raw(self, raw_item):
        raw_value = _np.array(raw_item, _np.uint8, copy=False)
        if self._fastpath == _FASTPATH_NUMPY:
            max_elements = raw_value.shape[0] // self._internal_dtype.itemsize
            shape = self.dynamic_shape(max_elements)
            elements = _shape_elements(shape)
            if elements > max_elements:
                raise ValueError('Item {} has too few elements for shape ({} < {})'.format(
                                 self.name, max_elements, elements))
            size_bytes = elements * self._internal_dtype.itemsize
            if raw_item.is_immediate:
                # Immediates get head padding instead of tail padding
                # For some reason, np.frombuffer doesn't work on memoryview, but np.array does
                array1d = raw_value[-size_bytes:]
            else:
                array1d = raw_value[:size_bytes]
            array1d = array1d.view(dtype=self._internal_dtype)
            # Force to native endian
            array1d = array1d.astype(self._internal_dtype.newbyteorder('='),
                                     casting='equiv', copy=False)
            value = _np.reshape(array1d, shape, self.order)
        elif (self._fastpath == _FASTPATH_IMMEDIATE and
                raw_item.is_immediate and
                raw_value.shape[0] * 8 == self.format[0][1]):
            value = raw_item.immediate_value
            if self.format[0][0] == 'i':
                top = 1 << (self.format[0][1] - 1)
                if value >= top:
                    value -= 2 * top
        else:
            itemsize_bits = self.itemsize_bits
            max_elements = raw_value.shape[0] * 8 // itemsize_bits
            shape = self.dynamic_shape(max_elements)
            elements = _shape_elements(shape)
            bits = elements * itemsize_bits
            if elements > max_elements:
                raise ValueError('Item {} has too few elements for shape ({} < {})'.format(
                                 self.name, max_elements, elements))
            if raw_item.is_immediate:
                # Immediates get head padding instead of tail padding
                size_bytes = (bits + 7) // 8
                raw_value = raw_value[-size_bytes:]

            gen = self._read_bits(raw_value)
            gen.send(None)    # Initialisation of the generator
            value = _np.array(self._load_recursive(shape, gen), self._internal_dtype)

        if len(self.shape) == 0 and isinstance(value, _np.ndarray):
            # Convert zero-dimensional array to scalar
            value = value[()]
        elif len(self.shape) == 1 and self.format == [('c', 8)]:
            # Convert array of characters to a string
            value = _bytes_to_str_ascii(b''.join(value))
        self.value = value

    def _num_elements(self):
        if isinstance(self.value, _np.ndarray):
            return self.value.size
        cur = self.value
        ans = 1
        for size in self.shape:
            ans *= len(cur)
            if ans == 0:
                return ans    # Prevents IndexError below
            cur = cur[0]
        return ans

    def _transform_value(self):
        """Mangle the value into a numpy array. This does several things:

        - If it is stringlike (bytes or unicode) and the expected shape is
          1D, it is split into an array of characters.
        - It is coerced to a numpy array, enforcing the dtype and order. Where
          possible, no copy is made.
        - The shape is checked against the expected shape.

        Returns
        -------
        value : :py:class:`numpy.ndarray`
            The transformed value

        Raises
        ------
        ValueError
            if the value is `None`
        ValueError
            if the value has the wrong shape
        TypeError
            if numpy raised it when trying to convert the value
        """
        value = self.value
        if value is None:
            raise ValueError('Cannot send a value of None')
        if (isinstance(value, (six.binary_type, six.text_type)) and
                len(self.shape) == 1):
            # This is complicated by Python 3 not providing a simple way to
            # turn a bytes object into a list of one-byte objects, the way
            # list(str) does.
            value = [self.value[i : i + 1] for i in range(len(self.value))]
        value = _np.array(value, dtype=self._internal_dtype, order=self.order, copy=False)
        if not self.compatible_shape(value.shape):
            raise ValueError('Value has shape {}, expected {}'.format(value.shape, self.shape))
        return value

    def to_buffer(self):
        """Returns an object that implements the buffer protocol for the value.
        It can be either the original value (on the numpy fast path), or a new
        temporary object.
        """
        value = self._transform_value()
        if self._fastpath != _FASTPATH_NUMPY:
            bit_length = self.itemsize_bits * self._num_elements()
            out = bytearray((bit_length + 7) // 8)
            gen = self._write_bits(out)
            gen.send(None)  # Initialise the generator
            # If it's a scalar, unpack it. That way, the input to the
            # final level of recursion in _store_recursive is always
            # the scalar rather than the 0D array.
            if len(self.shape) == 0:
                value = value[()]
            self._store_recursive(len(self.shape), value, gen)
            gen.close()
            return out
        else:
            if self.order == 'F':
                # numpy doesn't allow buffer protocol to be used on arrays that
                # aren't C-contiguous, but transposition just fiddles the
                # strides of the view without creating a new array.
                value = value.transpose()
            return value


class ItemGroup(object):
    """
    Items are collected into sets called *item groups*, which can be indexed by
    either item ID or item name.

    There are some subtleties with respect to re-issued item descriptors. There are
    two cases:

    1. The item descriptor is identical to a previous seen one. In this case, no
       action is taken.
    2. Otherwise, any existing items with the same name or ID (which could be two
       different items) are dropped, the new item is added, and its value
       becomes ``None``. The version is set to be higher than version on an item
       that was removed, so that consumers who only check the version will
       detect the change.
    """

    def __init__(self):
        self._by_id = {}
        self._by_name = {}

    def _remove_item(self, item):
        del self._by_id[item.id]
        del self._by_name[item.name]

    def _add_item(self, item):
        try:
            old = self._by_id[item.id]
        except KeyError:
            old = None
        try:
            old_by_name = self._by_name[item.name]
        except KeyError:
            old_by_name = None

        # Check if this is just the same thing
        if (old is not None and
                old.name == item.name and
                old.description == item.description and
                old.shape == item.shape and
                old.dtype == item.dtype and
                old.order == item.order and
                old.format == item.format):
            # Descriptor is the same, so just transfer the value. If the value
            # is None, then we've only been given a descriptor to add.
            if item.value is not None:
                old.value = item.value
            return

        if old is not None or old_by_name is not None:
            _logger.info('Descriptor replacement for ID %#x, name %s', item.id, item.name)
        # Ensure the version number is seen to increment, regardless of
        # whether accessed by name or ID.
        new_version = item.version
        if old is not None:
            new_version = max(new_version, old.version + 1)
        if old_by_name is not None:
            new_version = max(new_version, old_by_name.version + 1)
        item.version = new_version

        # Remove previous items, under the same name of ID
        if old is not None:
            self._remove_item(old)
        if old_by_name is not None and old_by_name is not old:
            self._remove_item(old_by_name)

        # Install new item
        self._by_id[item.id] = item
        self._by_name[item.name] = item

    def add_item(self, *args, **kwargs):
        """Add a new item to the group. The parameters are used to construct an
        :py:class:`Item`. If `id` is `None`, it will be automatically populated
        with an ID that is not already in use.

        See the class documentation for the behaviour when the name or ID
        collides with an existing one. In addition, if the item descriptor is
        identical to an existing one and a value, this value is assigned to
        the existing item.
        """
        item = Item(*args, **kwargs)
        if item.id is None:
            item.id = _UNRESERVED_ID
            while item.id in self._by_id:
                item.id += 1
        self._add_item(item)
        return item

    def __getitem__(self, key):
        """Dictionary-style lookup by either ID or name"""
        if isinstance(key, _numbers.Integral):
            return self._by_id[key]
        else:
            return self._by_name[key]

    def __contains__(self, key):
        """Dictionary-style membership test by either ID or name"""
        if isinstance(key, _numbers.Integral):
            return key in self._by_id
        else:
            return key in self._by_name

    def keys(self):
        """Item names"""
        return self._by_name.keys()

    def ids(self):
        """Item IDs"""
        return self._by_id.keys()

    def values(self):
        """Item values"""
        return self._by_name.values()

    def items(self):
        """Dictionary style (name, value) pairs"""
        return self._by_name.items()

    def __len__(self):
        """Number of items"""
        return len(self._by_name)

    def update(self, heap):
        """Update the item descriptors and items from an incoming heap.

        Parameters
        ----------
        heap : :class:`spead2.recv.Heap`
            Incoming heap

        Returns
        -------
        dict
            Items that have been updated from this heap, indexed by name
        """
        for descriptor in heap.get_descriptors():
            item = Item.from_raw(descriptor, flavour=heap.flavour)
            self._add_item(item)
        updated_items = {}
        for raw_item in heap.get_items():
            if raw_item.id <= STREAM_CTRL_ID:
                continue     # Special fields, not real items
            try:
                item = self._by_id[raw_item.id]
            except KeyError:
                _logger.warning('Item with ID %#x received but there is no descriptor', raw_item.id)
            else:
                item.set_from_raw(raw_item)
                item.version += 1
                updated_items[item.name] = item
        return updated_items
