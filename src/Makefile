CXX = g++
CXXFLAGS = -Wall -g -std=c++11 -D_GLIBCXX_DEBUG -fsanitize=address
LDFLAGS = -fsanitize=address -lboost_system
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
