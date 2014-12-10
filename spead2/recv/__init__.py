import atexit
import weakref
from _recv import *

class _StreamIter(object):
    def __init__(self, stream):
        self._stream = stream
        self._done = False

    def __iter__(self):
        return self

    def next(self):
        if not self._done:
            heap = self._stream.pop()
            if heap.cnt < 0:
                self._done = True
        if self._done:
            raise StopIteration
        else:
            return heap

    # For Python 3 compatibility
    def __next__(self):
        return self.next()

Stream.__iter__ = lambda x: _StreamIter(x)

# Set of weak references to receivers
_receiver_cleanups = set()

@atexit.register
def _stop_receivers():
    for ref in _receiver_cleanups:
        obj = ref()
        if obj is not None:
            obj.stop()
            obj.join()
    _receiver_cleanups.clear()

_Receiver = Receiver

class Receiver(_Receiver):
    def __init__(self):
        _Receiver.__init__(self)
        ref = weakref.ref(self, _receiver_cleanups.discard)
        _receiver_cleanups.add(ref)
