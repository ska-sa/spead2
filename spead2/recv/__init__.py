import atexit
import weakref
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
