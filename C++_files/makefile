CC = g++
CFLAGS = -g -Wall -pthread -D WITH_TBB=ON -D WITH_CSTRIPES=ON
SRCS = main.cpp
PROG = main

OPENCV = `pkg-config --cflags --libs opencv4`
LIBS = $(OPENCV)
LIBS += -L/home/pi/tensorflow_src/tensorflow/lite/tools/make/gen/rpi_armv7l/lib -ltensorflow-lite -ldl

INCS += -I/home/pi/tensorflow_src \
        -I/home/pi/tensorflow_src/tensorflow/lite/tools/make/downloads/flatbuffers/include \
	-I/home/pi/tensorflow_src/tensorflow/lite/tools/make/downloads/absl \
	-I/home/pi/eigen-3.3.7

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) $(INCS) -DOPENCV_GENERATE_PKGCONFIG=ON
