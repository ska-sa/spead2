VARIANT = release

CXX = g++
CXXFLAGS = -Wall -std=c++11 -pthread
LDFLAGS = -lboost_system -pthread
ifeq ($(VARIANT), fulldebug)
    CXXFLAGS += -g -fsanitize=address -D_GLIBCXX_DEBUG -DSPEAD_MAX_LOG_LEVEL=spead::log_level::debug
    LDFLAGS += -fsanitize=address
endif
ifeq ($(VARIANT), debug)
    CXXFLAGS += -g -ggdb
    LDFLAGS += -g
endif
ifeq ($(VARIANT), release)
    CXXFLAGS += -O3 -DNDEBUG
    LDFLAGS +=
endif

RECV_SOURCES = $(wildcard recv*.cpp)
RECV_OBJECTS = $(patsubst %.cpp, %.o, $(RECV_SOURCES))
COMMON_SOURCES = $(wildcard common*.cpp)
COMMON_OBJECTS = $(patsubst %.cpp, %.o, $(COMMON_SOURCES))
TARGETS = test_recv

test_recv_OBJECTS = test_recv.o $(RECV_OBJECTS) $(COMMON_OBJECTS)

all: $(TARGETS)

%.o: %.cpp *.h
	$(CXX) $(CXXFLAGS) -c $<

test_recv: $(test_recv_OBJECTS)
	$(CXX) $(test_recv_OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f *.o $(TARGETS)
