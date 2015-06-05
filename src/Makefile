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

RECV_SOURCES = $(wildcard recv*.cpp)
RECV_OBJECTS = $(patsubst %.cpp, %.o, $(RECV_SOURCES))
SEND_SOURCES = $(wildcard send*.cpp)
SEND_OBJECTS = $(patsubst %.cpp, %.o, $(SEND_SOURCES))
COMMON_SOURCES = $(wildcard common*.cpp)
COMMON_OBJECTS = $(patsubst %.cpp, %.o, $(COMMON_SOURCES))
TARGETS = libspead2.a test_recv test_send spead2_bench

all: $(TARGETS)

%.o: %.cpp *.h
	$(CXX) $(CXXFLAGS) -c $<

libspead2.a: $(RECV_OBJECTS) $(SEND_OBJECTS) $(COMMON_OBJECTS)
	$(AR) rcs $@ $(RECV_OBJECTS) $(SEND_OBJECTS) $(COMMON_OBJECTS)

test_recv: test_recv.o libspead2.a
	$(CXX) -o $@ test_recv.o -L. -lspead2 $(LDFLAGS)

test_send: test_send.o libspead2.a
	$(CXX) -o $@ test_send.o -L. -lspead2 $(LDFLAGS)

spead2_bench: spead2_bench.o libspead2.a
	$(CXX) -o $@ spead2_bench.o -L. -lspead2 $(LDFLAGS) -lboost_program_options

clean:
	rm -f *.o $(TARGETS)
