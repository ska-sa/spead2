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

    Parameters
    ----------
    item_group : :py:class:`spead2.ItemGroup`
        Item group to monitor.
    descriptor_frequency : int, optional
        If specified, descriptors will be re-sent once every `descriptor_frequency` heaps.
    heap_address_bits : int
        Determines the SPEAD flavour.
    bug_compat : int
        Bug compatibility (see :ref:`py-bug-compat`)
    """
    def __init__(self, item_group, descriptor_frequency=None,
                 heap_address_bits=Heap.DEFAULT_HEAP_ADDRESS_BITS, bug_compat=0):
        self._item_group = item_group
        self._info = {}              # Maps ID to _ItemInfo
        self._next_cnt = 1
        self._descriptor_frequency = descriptor_frequency
        self._heap_address_bits = heap_address_bits
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
        """Return a new heap which contains all the new items and item
        descriptors since the last call.
        """
        heap = Heap(self._next_cnt, self._heap_address_bits, self._bug_compat)
        for item in self._item_group.values():
            info = self._get_info(item)
            if self._descriptor_stale(item, info):
                heap.add_descriptor(item)
                info.descriptor_cnt = self._next_cnt
            if item.value is not None and info.version != item.version:
                heap.add_item(item)
                info.version = item.version
        self._next_cnt += 1
        return heap

    def get_end(self):
        """Return a heap that contains only an end-of-stream marker.
        """
        heap = Heap(self._next_cnt, self._heap_address_bits, self._bug_compat)
        heap.add_end()
        self._next_cnt += 1
        return heap


class ItemGroup(_spead2.ItemGroup, HeapGenerator):
    """Bundles an ItemGroup and HeapGenerator into a single class"""
    def __init__(self, *args, **kwargs):
        _spead2.ItemGroup.__init__(self)
        HeapGenerator.__init__(self, self, *args, **kwargs)
