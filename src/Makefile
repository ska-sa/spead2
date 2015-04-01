VARIANT = release

AR = gcc-ar
CXX = g++
CXXFLAGS = -Wall -std=c++11 -pthread
LDFLAGS = -lboost_system -pthread
ifeq ($(VARIANT), fulldebug)
    CXXFLAGS += -g -fsanitize=address -D_GLIBCXX_DEBUG -DSPEAD2_MAX_LOG_LEVEL=spead::log_level::debug
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
TARGETS = libspead2.a test_recv test_send

test_recv_OBJECTS = test_recv.o $(RECV_OBJECTS) $(COMMON_OBJECTS)
test_send_OBJECTS = test_send.o $(SEND_OBJECTS) $(COMMON_OBJECTS)

all: $(TARGETS)

%.o: %.cpp *.h
	$(CXX) $(CXXFLAGS) -c $<

libspead2.a: $(RECV_OBJECTS) $(SEND_OBJECTS) $(COMMON_OBJECTS)
	$(AR) rcs $@ $(RECV_OBJECTS) $(SEND_OBJECTS) $(COMMON_OBJECTS)

test_recv: test_recv.o libspead2.a
	$(CXX) -o $@ test_recv.o -L. -lspead2 $(LDFLAGS)

test_send: test_send.o libspead2.a
	$(CXX) -o $@ test_send.o -L. -lspead2 $(LDFLAGS)

clean:
	rm -f *.o $(TARGETS)
