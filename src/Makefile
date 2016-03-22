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

VARIANT = release

AR = gcc-ar
CXX = g++
CXXFLAGS = -Wall -std=c++11 -pthread
CXXFLAGS_UNITTEST = -DBOOST_TEST_DYN_LINK
LDFLAGS = -lboost_system -pthread
ifeq ($(VARIANT), fulldebug)
    CXXFLAGS += -g -fsanitize=address -D_GLIBCXX_DEBUG -DSPEAD2_MAX_LOG_LEVEL=spead2::log_level::debug
    LDFLAGS += -fsanitize=address
endif
ifeq ($(VARIANT), debug)
    CXXFLAGS += -g -ggdb
    LDFLAGS += -g
endif
ifeq ($(VARIANT), release)
    CXXFLAGS += -O3 -DNDEBUG -flto
    LDFLAGS += $(CXXFLAGS) -Wl,--no-as-needed
endif
ifeq ($(NETMAP),1)
    CXXFLAGS += -DSPEAD2_USE_NETMAP=1
endif
ifneq ($(RECVMMSG),)
    CXXFLAGS += -DSPEAD2_USE_RECVMMSG=$(RECVMMSG)
endif
ifneq ($(EVENTFD),)
    CXXFLAGS += -DSPEAD2_USE_EVENTFD=$(EVENTFD)
endif
ifneq ($(MOVNTDQ),)
    CXXFLAGS += -DSPEAD2_USE_MOVNTDQ=$(MOVNTDQ)
endif

RECV_SOURCES = $(wildcard recv_*.cpp)
RECV_OBJECTS = $(patsubst %.cpp, %.o, $(RECV_SOURCES))
SEND_SOURCES = $(wildcard send_*.cpp)
SEND_OBJECTS = $(patsubst %.cpp, %.o, $(SEND_SOURCES))
COMMON_SOURCES = $(wildcard common_*.cpp)
COMMON_OBJECTS = $(patsubst %.cpp, %.o, $(COMMON_SOURCES))
TEST_SOURCES = $(wildcard unittest_*.cpp)
TEST_OBJECTS = $(patsubst %.cpp, %.o, $(TEST_SOURCES))
TARGETS = libspead2.a test_recv test_send test_ringbuffer spead2_bench spead2_recv
TEST_TARGETS = spead2_unittest

all: $(TARGETS)

test: $(TEST_TARGETS)
	./spead2_unittest

unittest_%.o: unittest_%.cpp *.h
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_UNITTEST) -c $<

%.o: %.cpp *.h
	$(CXX) $(CXXFLAGS) -c $<

libspead2.a: $(RECV_OBJECTS) $(SEND_OBJECTS) $(COMMON_OBJECTS)
	$(AR) rcs $@ $(RECV_OBJECTS) $(SEND_OBJECTS) $(COMMON_OBJECTS)

test_recv: test_recv.o libspead2.a
	$(CXX) -o $@ test_recv.o -L. -lspead2 $(LDFLAGS)

test_recv_netmap: test_recv_netmap.o libspead2.a
	$(CXX) -o $@ test_recv_netmap.o -L. -lspead2 $(LDFLAGS)

test_send: test_send.o libspead2.a
	$(CXX) -o $@ test_send.o -L. -lspead2 $(LDFLAGS)

test_ringbuffer: test_ringbuffer.o libspead2.a
	$(CXX) -o $@ test_ringbuffer.o -L. -lspead2 $(LDFLAGS) -lboost_program_options

spead2_bench: spead2_bench.o libspead2.a
	$(CXX) -o $@ spead2_bench.o -L. -lspead2 $(LDFLAGS) -lboost_program_options

spead2_recv: spead2_recv.o libspead2.a
	$(CXX) -o $@ spead2_recv.o -L. -lspead2 $(LDFLAGS) -lboost_program_options

spead2_unittest: $(TEST_OBJECTS) libspead2.a
	$(CXX) -o $@ $(TEST_OBJECTS) -L. -lspead2 $(LDFLAGS) -lboost_unit_test_framework

clean:
	rm -f *.o $(TARGETS)

.PHONY: all clean test
