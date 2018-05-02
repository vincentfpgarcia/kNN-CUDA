.PHONY: all test

all: test

test:
	nvcc -o test test.cpp knncuda.cu -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets

clean:
	rm test
