IDIR = include
SDIR = src
BINDIR = bin
DATADIR = data

CXX=g++
CXXFLAGS= -I$(IDIR) -std=c++17 -g -O3  $(CV_FLAGS)
LDFLAGS = $(GMP) $(CV_LIBS)

CV_FLAGS = `pkg-config --cflags opencv4`
CV_LIBS = `pkg-config --libs opencv4`
GMP = -lgmp -lgmpxx

DEPS = $(IDIR)/$(wildcard *.hpp)

ODIR=obj
create_build_folder := $(shell mkdir -p $(ODIR))

OUT_DIR=output

_OBJ = $(patsubst %.cpp,%.o,$(wildcard $(SDIR)/*.cpp))
OBJ = $(patsubst $(SDIR)/%,$(ODIR)/%,$(_OBJ))

create_build_folder := $(shell mkdir -p $(BINDIR))
TARGET = $(BINDIR)/app

all: $(TARGET) run gif

$(TARGET): $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	rm -rf $(OUT_DIR)
	mkdir $(OUT_DIR)
	time $< $(input_file)

gif:
	ffmpeg -f image2 -framerate 60 -i $(OUT_DIR)/gif-%05d.png -loop -1 $(OUT_DIR)/_$(input_file:.png=.gif) 
	open $(OUT_DIR)

$(ODIR)/%.o: $(SDIR)/%.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean run gif

clean:
	rm -f $(ODIR)/*.o $(TARGET)