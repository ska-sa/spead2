VARIANT = debug

CXX = g++
CXXFLAGS = -Wall -std=c++11
LDFLAGS = -lboost_system
ifeq ($(VARIANT), fulldebug)
    CXXFLAGS += -g -D_GLIBCXX_DEBUG -fsanitize=address
    LDFLAGS += -fsanitize=address
endif
ifeq ($(VARIANT), debug)
    CXXFLAGS += -g -ggdb
    LDFLAGS += -g
endif
ifeq ($(VARIANT), release)
    CXXFLAGS += -O3 -flto -D_NDEBUG
    LDFLAGS += -flto
endif

SOURCES = $(wildcard *.cpp)
OBJECTS = $(patsubst %.cpp, %.o, $(SOURCES))
TARGETS = test_in

test_in_OBJECTS = in.o udp_in.o test_in.o

all: $(TARGETS)

%.o: %.cpp *.h
	$(CXX) $(CXXFLAGS) -c $<

test_in: $(test_in_OBJECTS)
	$(CXX) $(test_in_OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -f *.o $(TARGETS)
