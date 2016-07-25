INCLUDE = -I./include
LIBOPTS = -L./lib64
LDFLAGS := -ltensorflow
CFLAGS = -O3 -fpic -Wall -std=c++11
CC = g++

.PHONY : all
all : app

app :
	$(CC) app.cc $(CFLAGS) $(INCLUDE) $(LIBOPTS) -o $@ $(LDFLAGS)

.PHONY : clean
clean :
	rm -f *.o app
