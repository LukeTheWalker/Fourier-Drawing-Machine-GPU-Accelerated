CXX=g++
CXXFLAGS= -I. -std=c++17 -g -O3  $(CV_FLAGS)
DEPS = $(wildcard *.hpp)
LDFLAGS = $(GMP) $(CV_LIBS)

CV_FLAGS = `pkg-config --cflags opencv4`
CV_LIBS = `pkg-config --libs opencv4`
GMP = -lgmp -lgmpxx

ODIR=obj
create_build_folder := $(shell mkdir -p $(ODIR))

OUT_DIR=output

_OBJ = $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: app run gif

app: $(OBJ)
	$(CXX) -o $@ $^ $(LDFLAGS)

run: app
	rm -rf $(OUT_DIR)
	mkdir $(OUT_DIR)
	time ./app $(input_file)

gif:
	ffmpeg -f image2 -framerate 60 -i $(OUT_DIR)/gif-%05d.png -loop -1 $(OUT_DIR)/_$(input_file:.png=.gif) 
	open $(OUT_DIR)

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean run gif

clean:
	rm -f $(ODIR)/*.o app