INCLUDE = -I./include
LIBOPTS = -L./lib
LDFLAGS := -ltensorflow
CFLAGS = -O3 -fpic -Wall -std=c++11
CC = g++

.PHONY : all
all : run

dependency :
	echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
	curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
	sudo apt-get update && sudo apt-get -y install bazel python-pip python-numpy swig python-dev python-wheel python-wheel-common

build :
	rm -rf build && mkdir -p build
	cd build && cmake .. && make

data :
	mkdir -p data
	cd data && wget https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
	cd data && wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/grace_hopper.jpg
	cd data && unzip inception_dec_2015.zip
	rm -f data/inception_dec_2015.zip

app : build data lib/libtensorflow.so
	$(CC) app.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

run : app
	LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:./lib && CUDA_VISIBLE_DEVICES=0 ./app ./data/grace_hopper.jpg

clean :
	rm -f *.o app

reset : clean
	rm -rf build include data lib/libtensorflow.so lib/tensorflow-*
