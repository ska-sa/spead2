AM_CXXFLAGS = -std=c++11
AM_CPPFLAGS = -I$(top_builddir)/include -I$(top_srcdir)/include
LDADD = libspead2.a -lboost_system -lpthread
# We use common_features.h to pass feature settings, not defines or config.h
DEFS =
AM_LDFLAGS =

if DEBUG_SYMBOLS
AM_CXXFLAGS += -g
else
AM_LDFLAGS += -s
endif

if DEBUG_LOG
DEFS += -DSPEAD2_MAX_LOG_LEVEL=spead2::log_level::debug
endif

if OPTIMIZED
AM_CXXFLAGS += -O3
DEFS += -DNDEBUG
if HAVE_LTO
AM_CXXFLAGS += -flto
AM_LDFLAGS += $(AM_CXXFLAGS) -Wl,--no-as-needed
endif
else
AM_CXXFLAGS += -O0
DEFS += -DDEBUG
endif

if SPEAD2_USE_IBV
LIBS += -lrdmacm -libverbs
endif

lib_LIBRARIES = libspead2.a
bin_PROGRAMS = spead2_recv spead2_send spead2_bench
noinst_PROGRAMS = test_recv test_send
check_PROGRAMS = spead2_unittest
TESTS = spead2_unittest

spead2_recv_SOURCES = spead2_recv.cpp
spead2_recv_LDADD = -lboost_program_options $(LDADD)

spead2_send_SOURCES = spead2_send.cpp
spead2_send_LDADD = -lboost_program_options $(LDADD)

spead2_bench_SOURCES = spead2_bench.cpp
spead2_bench_LDADD = -lboost_program_options $(LDADD)

test_recv_SOURCES = test_recv.cpp

test_send_SOURCES = test_send.cpp

spead2_unittest_SOURCES = \
	unittest_main.cpp \
	unittest_memcpy.cpp \
	unittest_memory_allocator.cpp \
	unittest_memory_pool.cpp \
	unittest_recv_live_heap.cpp
spead2_unittest_CPPFLAGS = -DBOOST_TEST_DYN_LINK $(AM_CPPFLAGS)
spead2_unittest_LDADD = -lboost_unit_test_framework $(LDADD)

libspead2_a_SOURCES = \
	common_flavour.cpp \
	common_ibv.cpp \
	common_logging.cpp \
	common_memcpy.cpp \
	common_memory_allocator.cpp \
	common_memory_pool.cpp \
	common_raw_packet.cpp \
	common_semaphore.cpp \
	common_thread_pool.cpp \
	recv_heap.cpp \
	recv_live_heap.cpp \
	recv_mem.cpp \
	recv_netmap.cpp \
	recv_packet.cpp \
	recv_reader.cpp \
	recv_ring_stream.cpp \
	recv_stream.cpp \
	recv_udp_base.cpp \
	recv_udp.cpp \
	recv_udp_ibv.cpp \
	send_heap.cpp \
	send_packet.cpp \
	send_streambuf.cpp \
	send_stream.cpp \
	send_udp.cpp \
	send_udp_ibv.cpp
