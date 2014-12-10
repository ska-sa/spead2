import atexit
import weakref
import numpy.lib.utils
import numpy as np
import spead2
from _recv import *

# Set of weak references to receivers
_receiver_cleanups = set()

@atexit.register
def _stop_receivers():
    for ref in _receiver_cleanups:
        obj = ref()
        if obj is not None:
            try:
                obj.stop()
            except ValueError:
                pass  # Can happen if obj was not running yet
    _receiver_cleanups.clear()

_Receiver = Receiver

class Receiver(_Receiver):
    # TODO: do this only on start()?
    def __init__(self):
        _Receiver.__init__(self)
        ref = weakref.ref(self, _receiver_cleanups.discard)
        _receiver_cleanups.add(ref)

class Descriptor(object):
    @classmethod
    def _parse_numpy_descriptor(cls, descriptor_string):
        try:
            d = numpy.lib.utils.safe_eval(descriptor_string)
        except SyntaxError, e:
            msg = "Cannot parse descriptor: %r\nException: %r"
            raise ValueError(msg % (descriptor_string, e))
        if not isinstance(d, dict):
            msg = "Descriptor is not a dictionary: %r"
            raise ValueError(msg % d)
        keys = d.keys()
        keys.sort()
        if keys != ['descr', 'fortran_order', 'shape']:
            msg = "Descriptor does not contain the correct keys: %r"
            raise ValueError(msg % (keys,))
        # Sanity-check the values.
        if not isinstance(d['shape'], tuple) or not all([isinstance(x, (int, long)) for x in d['shape']]):
            msg = "shape is not valid: %r"
            raise ValueError(msg % (d['shape'],))
        if not isinstance(d['fortran_order'], bool):
            msg = "fortran_order is not a valid bool: %r"
            raise ValueError(msg % (d['fortran_order'],))
        try:
            dtype = numpy.dtype(d['descr'])
        except TypeError, e:
            msg = "descr is not a valid dtype descriptor: %r"
            raise ValueError(msg % (d['descr'],))
        return d['shape'], d['fortran_order'], dtype

    def __init__(self, raw_descriptor):
        self.id = raw_descriptor.id
        self.name = raw_descriptor.name
        self.description = raw_descriptor.description
        if raw_descriptor.dtype:
            self.shape, self.fortran_order, self.dtype = \
                    self._parse_numpy_descriptor(raw_descriptor.dtype)
            if spead2.BUG_COMPAT_SWAP_ENDIAN:
                self.dtype = self.dtype.newbyteorder()
        else:
            self.shape = []
            self.fortran_order = False
            self.dtype = None

class Item(Descriptor):
    def __init__(self, *args, **kw):
        super(Item, self).__init__(*args, **kw)
        self.value = None

    def set_from_raw(self, raw_item):
        elements = int(np.product(self.shape))
        raw_value = raw_item.value
        if self.dtype is None or not isinstance(raw_value, memoryview):
            self.value = raw_value
        else:
            if raw_value.shape[0] < elements * self.dtype.itemsize:
                raise TypeError('Item has too few elements (%d < %d)' % (raw_value.shape[0], elements))
            # For some reason, np.frombuffer doesn't work on memoryview, but np.array does
            array1d = np.array(raw_item.value, copy=False).view(dtype=self.dtype)[:elements]
            if self.dtype.byteorder in ('<', '>'):
                # Either < or > indicates non-native endianness. Swap it now
                # so that calculations later will be efficient
                dtype = self.dtype.newbyteorder()
                array1d = array1d.byteswap(True).view(dtype=dtype)
            order = 'F' if self.fortran_order else 'C'
            self.value = np.reshape(array1d, self.shape, order)

class ItemGroup(object):
    def __init__(self):
        self.items = {}

    def update(self, heap):
        for descriptor in heap.get_descriptors():
            self.items[descriptor.id] = Item(descriptor)
        for raw_item in heap.get_items():
            try:
                item = self.items[raw_item.id]
            except KeyError:
                # TODO: log it
                pass
            else:
                item.set_from_raw(raw_item)
