# Copyright 2016 SKA South Africa
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

include $(srcdir)/Makefile.inc.am

lib_LIBRARIES = libspead2.a
bin_PROGRAMS = spead2_recv spead2_send spead2_bench
check_PROGRAMS = spead2_unittest
TESTS = spead2_unittest

if SPEAD2_USE_IBV
bin_PROGRAMS += mcdump
mcdump_SOURCES = mcdump.cpp
mcdump_LDADD = -lboost_program_options $(LDADD)
endif

spead2_recv_SOURCES = spead2_recv.cpp
spead2_recv_LDADD = -lboost_program_options $(LDADD)

spead2_send_SOURCES = spead2_send.cpp
spead2_send_LDADD = -lboost_program_options $(LDADD)

spead2_bench_SOURCES = spead2_bench.cpp
spead2_bench_LDADD = -lboost_program_options $(LDADD)

spead2_unittest_SOURCES = \
	unittest_main.cpp \
	unittest_logging.cpp \
	unittest_memcpy.cpp \
	unittest_memory_allocator.cpp \
	unittest_memory_pool.cpp \
	unittest_raw_packet.cpp \
	unittest_recv_live_heap.cpp \
	unittest_recv_custom_memcpy.cpp \
	unittest_semaphore.cpp \
	unittest_send_heap.cpp \
	unittest_send_streambuf.cpp
spead2_unittest_CPPFLAGS = -DBOOST_TEST_DYN_LINK $(AM_CPPFLAGS)
spead2_unittest_LDADD = -lboost_unit_test_framework $(LDADD)

libspead2_a_SOURCES = \
	common_flavour.cpp \
	common_ibv.cpp \
	common_ibv_loader.cpp \
	common_ibv_loader_utils.cpp \
	common_inproc.cpp \
	common_logging.cpp \
	common_memcpy.cpp \
	common_memory_allocator.cpp \
	common_memory_pool.cpp \
	common_raw_packet.cpp \
	common_semaphore.cpp \
	common_socket.cpp \
	common_thread_pool.cpp \
	recv_heap.cpp \
	recv_inproc.cpp \
	recv_live_heap.cpp \
	recv_mem.cpp \
	recv_netmap.cpp \
	recv_packet.cpp \
	recv_reader.cpp \
	recv_ring_stream.cpp \
	recv_stream.cpp \
	recv_tcp.cpp \
	recv_udp_base.cpp \
	recv_udp.cpp \
	recv_udp_ibv.cpp \
	recv_udp_ibv_mprq.cpp \
	recv_udp_pcap.cpp \
	send_heap.cpp \
	send_inproc.cpp \
	send_packet.cpp \
	send_streambuf.cpp \
	send_stream.cpp \
	send_tcp.cpp \
	send_udp.cpp \
	send_udp_ibv.cpp
