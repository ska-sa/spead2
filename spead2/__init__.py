import spead2._spead2
from spead2._spead2 import ThreadPool, MemPool
from spead2._spead2 import BUG_COMPAT_DESCRIPTOR_WIDTHS, BUG_COMPAT_SHAPE_BIT_1, BUG_COMPAT_SWAP_ENDIAN
import numbers as _numbers
import numpy as _np

BUG_COMPAT_PYSPEAD_0_5_2 = BUG_COMPAT_DESCRIPTOR_WIDTHS | BUG_COMPAT_SHAPE_BIT_1 | BUG_COMPAT_SWAP_ENDIAN

class Descriptor(object):
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
        if not isinstance(d['shape'], tuple) or not all([isinstance(x, _numbers.Integral) for x in d['shape']]):
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
        return d['shape'], d['fortran_order'], dtype

    @classmethod
    def _make_numpy_header(self, shape, dtype, fortran_order):
        return "{{'descr': {!r}, 'fortran_order': {!r}, 'shape': {!r}}}".format(
                _np.lib.format.dtype_to_descr(dtype), fortran_order,
                tuple(shape))

    @classmethod
    def _parse_format(cls, fmt):
        """Attempt to convert a SPEAD format specification to a numpy dtype.
        If there is an unsupported field, returns None.
        """
        fields = []
        for code, length in fmt:
            if ( (code in ('u', 'i') and length in (8, 16, 32, 64)) or
                (code == 'f' and length in (32, 64)) or
                (code == 'b' and length == 8) ):
                fields.append('>' + code + str(length // 8))
            elif code == 'c' and length == 8:
                fields.append('S1')
            else:
                return None
        return _np.dtype(','.join(fields))

    def dynamic_shape(self, max_elements):
        known = 1
        unknown_pos = -1
        for i, x in enumerate(self.shape):
            if x >= 0:
                known *= x
            elif unknown_pos >= 0:
                raise TypeError('Shape has multiple unknown dimensions')
            else:
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

    def __init__(self, id, name, description, shape, dtype, order='C', format=None):
        if dtype is not None:
            dtype = _np.dtype(dtype)
            if format is not None:
                raise ValueError('Only one of dtype and format can be specified')
        else:
            if format is None:
                raise ValueError('One of dtype and format must be specified')
            if order != 'C':
                raise ValueError("When specifying format, order must be 'C'")
        if order not in ['C', 'F']:
            raise ValueError("Order must be 'C' or 'F'")
        self.id = id
        self.name = name
        self.description = description
        self.shape = shape
        self.dtype = dtype
        self.fortran_order = (order == 'F')
        self.format = format

    @classmethod
    def from_raw(cls, raw_descriptor, bug_compat):
        format = None
        if raw_descriptor.numpy_header:
            shape, fortran_order, dtype = \
                    cls._parse_numpy_header(raw_descriptor.numpy_header)
            if bug_compat & BUG_COMPAT_SWAP_ENDIAN:
                dtype = dtype.newbyteorder()
        else:
            shape = raw_descriptor.shape
            fortran_order = False
            dtype = cls._parse_format(raw_descriptor.format)
            if dtype is None:
                format = raw_descriptor.format
        return cls(
                raw_descriptor.id,
                raw_descriptor.name,
                raw_descriptor.description,
                shape, dtype, 'F' if fortran_order else 'C', format)

    def to_raw(self, bug_compat):
        raw = spead2._spead2.RawDescriptor()
        raw.id = self.id
        raw.name = self.name
        raw.description = self.description
        raw.shape = self.shape
        if self.dtype is not None:
            if bug_compat & BUG_COMPAT_SWAP_ENDIAN:
                dtype = self.dtype.newbyteorder()
            else:
                dtype = self.dtype
            raw.numpy_header = self._make_numpy_header(self.shape, dtype, self.fortran_order)
        else:
            raw.format = self.format
        return raw

class Item(Descriptor):
    def __init__(self, *args, **kw):
        super(Item, self).__init__(*args, **kw)
        self.value = None

    @classmethod
    def _bit_generator(cls, raw_value):
        """Generator that takes a memory view and provides bitfields from it.
        After creating the generator, call `send(None)` to initialise it, and
        thereafter call `send(need_bits)` to obtain that many bits.
        over the bits in raw_value (which must implement the buffer protocol),
        with the bits in each byte enumerated high-to-low.
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
            result = bits >> (have_bits - need_bits)
            bits &= (1 << (have_bits - need_bits)) - 1
            have_bits -= need_bits

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
                    raise ValueError('unhandled format {0}'.format((code, length)))
                fields.append(field)
            if len(fields) == 1:
                ans = fields[0]
            else:
                ans = tuple(fields)
        return ans

    def set_from_raw(self, raw_item):
        raw_value = raw_item.value
        if self.dtype is None:
            bit_length = 0
            for code, length in self.format:
                bit_length += length
            max_elements = raw_value.shape[0] * 8 // bit_length
            shape = self.dynamic_shape(max_elements)
            elements = int(_np.product(shape))
            if elements > max_elements:
                raise ValueError('Item has too few elements for shape (%d < %d)' % (max_elements, elements))

            gen = self._bit_generator(raw_value)
            gen.send(None) # Initialisation of the generator
            self.value = _np.array(self._load_recursive(shape, gen))
        else:
            max_elements = raw_value.shape[0] // self.dtype.itemsize
            shape = self.dynamic_shape(max_elements)
            elements = int(_np.product(shape))
            if elements > max_elements:
                raise ValueError('Item has too few elements for shape (%d < %d)' % (max_elements, elements))
            # For some reason, np.frombuffer doesn't work on memoryview, but np.array does
            array1d = _np.array(raw_value, copy=False)[: (elements * self.dtype.itemsize)]
            array1d = array1d.view(dtype=self.dtype)
            if self.dtype.byteorder in ('<', '>'):
                # Either < or > indicates non-native endianness. Swap it now
                # so that calculations later will be efficient
                dtype = self.dtype.newbyteorder()
                array1d = array1d.byteswap(True).view(dtype=dtype)
            order = 'F' if self.fortran_order else 'C'
            value = _np.reshape(array1d, self.shape, order)
            if len(self.shape) == 0:
                # Convert zero-dimensional array to scalar
                value = value[()]
            elif len(self.shape) == 1 and self.dtype == _np.dtype('S1'):
                # Convert array of characters to a string
                value = b''.join(value).decode('ascii')
            self.value = value

    def to_buffer(self):
        """Returns an object that implements the buffer protocol for the value.
        It can be either the original value (if the descriptor uses numpy
        protocol), or a new temporary object.
        """
        if self.dtype is not None:
            return self.value
        else:
            raise NotImplementedError('Non-numpy items can not yet be sent')
