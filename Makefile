SHELL := /bin/zsh

IDIR = include
SDIR = src
BINDIR = bin

ifeq ($(shell which nvcc),)	
	CXX := g++
else
	CXX := nvcc
	CUFLAGS := -arch=sm_86
endif

CXXFLAGS= -I$(IDIR) -std=c++17 -g -O3 $(CUFLAGS) $(CV_FLAGS)
LDFLAGS = $(GMP) $(CV_LIBS)

CV_FLAGS = `pkg-config --cflags opencv4`
CV_LIBS = `pkg-config --libs opencv4`
GMP = -lgmp -lgmpxx

DATADIR = data
# create_data_folder := $(shell mkdir -p $(DATADIR))

ODIR=obj
# create_build_folder := $(shell mkdir -p $(ODIR))

OUT_DIR=output

DEPS = $(IDIR)/$(wildcard *.hpp *.cuh)

_CUFILES = $(wildcard $(SDIR)/*.cu)
_CXXFILES = $(wildcard $(SDIR)/*.cpp)

CUFILES = $(notdir $(_CUFILES))
CXXFILES = $(notdir $(_CXXFILES))

_OBJ = $(_CUFILES:.cu=.o) $(_CXXFILES:.cpp=.o)
OBJ = $(patsubst $(SDIR)/%,$(ODIR)/%,$(_OBJ))

# create_bin_folder := $(shell mkdir -p $(BINDIR))
TARGET = $(BINDIR)/app

UNAME_S := $(shell uname -s)

input_file = $(DATADIR)/piccione.png

file_name = $(notdir $(input_file))
gif_file = $(OUT_DIR)/_$(basename $(file_name)).gif

$(TARGET): $(OBJ) | $(BINDIR)
	$(CXX) -o $@ $^ $(LDFLAGS)

all: $(TARGET) run gif

run: $(TARGET) | $(DATADIR)
	rm -rf $(OUT_DIR)
	mkdir $(OUT_DIR)
	time $< $(input_file)

gif:
	ffmpeg -f image2 -framerate 60 -i $(OUT_DIR)/gif-%05d.png -loop -1 $(gif_file) -hide_banner -loglevel error
ifeq ($(UNAME_S),Darwin)
	open $(OUT_DIR)
endif

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS) | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cu $(DEPS)  | $(ODIR)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean run gif

clean:
	rm -f $(ODIR)/*.o $(TARGET)

$(ODIR):
	mkdir -p $(ODIR)

$(BINDIR): 
	mkdir -p $(BINDIR)

$(DATADIR):
	mkdir -p $(DATADIR)