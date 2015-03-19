"""Send SPEAD protocol
"""

from __future__ import print_function, division
import spead2 as _spead2
import weakref
from spead2._send import StreamConfig, BytesStream, UdpStream, Heap, PacketGenerator


class _ItemInfo(object):
    def __init__(self, item):
        self.version = None
        self.descriptor_cnt = None
        self.item = weakref.ref(item)


class HeapGenerator(object):
    """Tracks which items and item values have previously been sent and
    generates delta heaps with sequential numbering.
    """
    def __init__(self, ig, descriptor_frequency=None, bug_compat=0):
        self._ig = ig
        self._info = {}              # Maps ID to _ItemInfo
        self._next_cnt = 1
        self._descriptor_frequency = descriptor_frequency
        self._bug_compat = bug_compat

    def _get_info(self, item):
        if item.id not in self._info:
            self._info[item.id] = _ItemInfo(item)
        return self._info[item.id]

    def _descriptor_stale(self, item, info):
        if info.descriptor_cnt is None:
            # Never been sent before
            return True
        if self._descriptor_frequency is not None \
                and self._next_cnt - info.descriptor_cnt >= self._descriptor_frequency:
            # This descriptor is due for a resend
            return True
        # Check for complete replacement of the item
        orig_item = info.item()
        if orig_item is not item:
            info.version = None
            info.item = weakref.ref(item)
            return True
        return False

    def get_heap(self):
        heap = Heap(self._next_cnt, self._bug_compat)
        for item in self._ig.values():
            info = self._get_info(item)
            if self._descriptor_stale(item, info):
                heap.add_descriptor(item)
                info.descriptor_cnt = self._next_cnt
            if item.value is not None and info.version != item.version:
                heap.add_item(item)
                info.version = item.version
        self._next_cnt += 1
        return heap


class ItemGroup(_spead2.ItemGroup, HeapGenerator):
    """Bundles an ItemGroup and HeapGenerator into a single class"""
    def __init__(self, descriptor_frequency=None, bug_compat=0):
        _spead2.ItemGroup.__init__(self)
        HeapGenerator.__init__(self, self, descriptor_frequency, bug_compat)
