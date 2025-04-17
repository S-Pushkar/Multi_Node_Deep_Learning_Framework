# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -fopenmp
LDFLAGS = -lyaml-cpp

# Source files
SRCS = main.cpp neural_network.cpp
OBJS = $(SRCS:.cpp=.o)

# Output binary
TARGET = deep_learning_framework

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run with a config file: make run CONFIG=config.yml
run: $(TARGET)
	./$(TARGET) $(CONFIG)

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)
