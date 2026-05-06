CXX		:= g++
CXX_FLAGS	:= -Wall -Wextra -std=c++17 -ggdb

BIN 	:= bin
SRC	:= src
EXECUTABLE := main

all: $(BIN)/$(EXECUTABLE)

run: clean all
	@echo "Running.."
	./$(BIN)/$(EXECUTABLE)

$(BIN)/$(EXECUTABLE): $(SRC)/*.cpp
	@echo "Building.."
	$(CXX) $(CXX_FLAGS) $^ -o $@

clean:
	@echo "Cleaning.."
	-rm -rf $(BIN)/*
