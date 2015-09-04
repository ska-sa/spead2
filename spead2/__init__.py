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

import spead2._spead2
from spead2._spead2 import Flavour, ThreadPool, MemoryPool, Stopped, Empty, Stopped, \
    BUG_COMPAT_DESCRIPTOR_WIDTHS, \
    BUG_COMPAT_SHAPE_BIT_1, \
    BUG_COMPAT_SWAP_ENDIAN, \
    BUG_COMPAT_PYSPEAD_0_5_2
import numbers as _numbers
import numpy as _np
import logging
from spead2._version import __version__


_logger = logging.getLogger(__name__)
_UNRESERVED_ID = 0x1000      #: First ID that can be auto-allocated


class Descriptor(object):
    """Metadata for a SPEAD item.

    There are a number of restrictions in the way the parameters combine,
    which will cause `ValueError` to be raised if violated:

    - At most one element of `shape` can be `None`.
    - Exactly one of `dtype` and `format` must be non-`None`.
    - If `dtype` is specified, `shape` cannot have any unknown dimensions.
    - If `format` is specified, `order` must be 'C'

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
    dtype : numpy data type
        Data type, or `None` if `format` will be used instead
    order : {'C', 'F'}
        Indicates C-order or Fortran-order storage
    format : list of pairs, optional
        Structure fields for generic (non-numpy) type. Each element of the list
        is a tuple of field code and bit length.
    """

    def __init__(self, id, name, description, shape, dtype, order='C', format=None):
        shape = tuple(shape)
        unknowns = sum([x is None for x in shape])
        if unknowns > 1:
            raise ValueError('Cannot have multiple unknown dimensions')
        if dtype is not None:
            dtype = _np.dtype(dtype)
            if format is not None:
                raise ValueError('Only one of dtype and format can be specified')
            if unknowns > 0:
                raise ValueError('Cannot have unknown dimensions when using numpy descriptor')
        else:
            if format is None:
                raise ValueError('One of dtype and format must be specified')
            if order != 'C':
                raise ValueError("When specifying format, order must be 'C'")
            # Try to find a compatible numpy format
            if unknowns == 0:
                dtype = self._parse_format(format)
            else:
                dtype = None
            if dtype is not None:
                format = None

        if order not in ['C', 'F']:
            raise ValueError("Order must be 'C' or 'F'")
        self.id = id
        self.name = name
        self.description = description
        self.shape = shape
        self.dtype = dtype
        self.order = order
        self.format = format

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
        except TypeError as e:
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
    def _parse_format(cls, fmt, fallback=False):
        """Attempt to convert a SPEAD format specification to a numpy dtype.
        If there is an unsupported field, return `None` if `fallback` is
        `False`, or use `O` if `fallback` is `True`.

        Raises
        ------
        ValueError
            If the format is illegal
        """
        fields = []
        if not fmt:
            raise ValueError('empty format')
        for code, length in fmt:
            if length == 0:
                raise ValueError('zero-length field (bug_compat mismatch?)')
            if ((code in ('u', 'i') and length in (8, 16, 32, 64)) or
                    (code == 'f' and length in (32, 64)) or
                    (code == 'b' and length == 8)):
                fields.append('>' + code + str(length // 8))
            elif code == 'c' and length == 8:
                fields.append('S1')
            elif fallback:
                if code not in ['u', 'i', 'c', 'b']:
                    raise ValueError('illegal format ({}, {})'.format(code, length))
                fields.append('O')
            else:
                return None
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
            shape, order, dtype = \
                    cls._parse_numpy_header(raw_descriptor.numpy_header)
            if flavour.bug_compat & BUG_COMPAT_SWAP_ENDIAN:
                dtype = dtype.newbyteorder()
        else:
            shape = raw_descriptor.shape
            order = 'C'
            format = raw_descriptor.format
        return cls(
                raw_descriptor.id,
                raw_descriptor.name,
                raw_descriptor.description,
                shape, dtype, order, format)

    def to_raw(self, flavour):
        raw = spead2._spead2.RawDescriptor()
        raw.id = self.id
        raw.name = self.name
        raw.description = self.description
        raw.shape = self.shape
        if self.dtype is not None:
            if flavour.bug_compat & BUG_COMPAT_SWAP_ENDIAN:
                dtype = self.dtype.newbyteorder()
            else:
                dtype = self.dtype
            raw.numpy_header = self._make_numpy_header(self.shape, dtype, self.order)
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
        byte_source = iter(bytearray(raw_value))
        result = 0
        while True:
            need_bits = yield result
            while have_bits < need_bits:
                try:
                    bits = (bits << 8) | next(byte_source)
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
                    field = chr(raw)
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

    def _store_recursive(self, expected_shape, value, gen):
        if len(expected_shape) > 0:
            if expected_shape[0] is not None and expected_shape[0] != len(value):
                raise ValueError('Value does not conform to the expected shape')
            for sub in value:
                self._store_recursive(expected_shape[1:], sub, gen)
        else:
            if isinstance(value, list):
                raise ValueError('Value has too many dimensions for shape')
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
                    # In Python 3, indexing a bytes object gives numerical
                    # values instead of a substring.
                    if isinstance(field, int):
                        raw = field
                    else:
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
        raw_value = raw_item.value
        if self.dtype is None:
            itemsize_bits = self.itemsize_bits
            max_elements = raw_value.shape[0] * 8 // itemsize_bits
            shape = self.dynamic_shape(max_elements)
            elements = int(_np.product(shape))
            bits = elements * itemsize_bits
            if elements > max_elements:
                raise ValueError('Item has too few elements for shape (%d < %d)' %
                                 (max_elements, elements))
            if raw_item.is_immediate:
                # Immediates get head padding instead of tail padding
                size_bytes = (bits + 7) // 8
                raw_value = raw_value[-size_bytes:]

            gen = self._read_bits(raw_value)
            gen.send(None)    # Initialisation of the generator
            value = _np.array(self._load_recursive(shape, gen),
                              dtype=self._parse_format(self.format, True))
        else:
            max_elements = raw_value.shape[0] // self.dtype.itemsize
            shape = self.dynamic_shape(max_elements)
            elements = int(_np.product(shape))
            if elements > max_elements:
                raise ValueError('Item has too few elements for shape (%d < %d)' %
                                 (max_elements, elements))
            size_bytes = elements * self.dtype.itemsize
            if raw_item.is_immediate:
                # Immediates get head padding instead of tail padding
                # For some reason, np.frombuffer doesn't work on memoryview, but np.array does
                array1d = _np.array(raw_value, copy=False)[-size_bytes:]
            else:
                array1d = _np.array(raw_value, copy=False)[:size_bytes]
            array1d = array1d.view(dtype=self.dtype)
            if self.dtype.byteorder in ('<', '>'):
                # Either < or > indicates non-native endianness. Swap it now
                # so that calculations later will be efficient
                dtype = self.dtype.newbyteorder()
                array1d = array1d.byteswap(True).view(dtype=dtype)
            value = _np.reshape(array1d, self.shape, self.order)

        if len(self.shape) == 0:
            # Convert zero-dimensional array to scalar
            value = value[()]
        elif len(self.shape) == 1 and value.dtype == _np.dtype('S1'):
            # Convert array of characters to a string
            value = b''.join(value).decode('ascii')
        self.value = value

    def _num_elements(self):
        if isinstance(self.value, _np.ndarray):
            return int(_np.product(self.value.shape))
        cur = self.value
        ans = 1
        for size in self.shape:
            ans *= len(cur)
            if ans == 0:
                return ans    # Prevents IndexError below
            cur = cur[0]
        return ans

    def to_buffer(self):
        """Returns an object that implements the buffer protocol for the value.
        It can be either the original value (if the descriptor uses numpy
        protocol), or a new temporary object.
        """
        if self.value is None:
            raise ValueError('Cannot send a value of None')
        if self.dtype is None:
            bit_length = self.itemsize_bits * self._num_elements()
            out = bytearray((bit_length + 7) // 8)
            gen = self._write_bits(out)
            gen.send(None)  # Initialise the generator
            self._store_recursive(self.shape, self.value, gen)
            gen.close()
            return out
        else:
            a = _np.array(self.value, dtype=self.dtype, order=self.order, copy=False)
            if not self.compatible_shape(a.shape):
                raise ValueError('Value has shape {}, expected {}'.format(a.shape, self.shape))
            if self.order == 'F':
                # numpy doesn't allow buffer protocol to be used on arrays that
                # aren't C-contiguous, but transposition just fiddles the
                # strides without creating a new array
                a = a.transpose()
            return a


class ItemGroup(object):
    def __init__(self):
        self._by_id = {}
        self._by_name = {}

    def _add_item(self, item):
        if item.id in self._by_id or item.name in self._by_name:
            _logger.info('Descriptor replacement for ID %d, name %s', item.id, item.name)
        self._by_id[item.id] = item
        self._by_name[item.name] = item

    def add_item(self, *args, **kwargs):
        """Add a new item to the group. The parameters are used to construct an
        :py:class:`Item`. If `id` is `None`, it will be automatically populated
        with an ID that is not already in use.
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
            if raw_item.id <= 6:
                continue     # Special fields, not real items
            try:
                item = self._by_id[raw_item.id]
            except KeyError:
                _logger.warning('Item with ID %d received but there is no descriptor', raw_item.id)
            else:
                item.set_from_raw(raw_item)
                item.version = heap.cnt
                updated_items[item.name] = item
        return updated_items
