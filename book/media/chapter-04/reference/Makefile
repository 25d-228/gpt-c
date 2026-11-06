CC = xcrun clang
CPPFLAGS = -Iinclude
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2
LDLIBS = -lm
SOURCES = src/common.c src/tensor.c src/kernels.c tests/test_kernels.c
HEADERS = include/dl/common.h include/dl/tensor.h include/dl/kernels.h

.PHONY: all debug sanitize release test clean

all: debug

debug: build/debug/test-kernels
sanitize: build/sanitize/test-kernels
release: build/release/test-kernels

build/debug/test-kernels: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build/debug
	$(CC) $(CPPFLAGS) $(DEBUG_FLAGS) $(SOURCES) -o $@ $(LDLIBS)

build/sanitize/test-kernels: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(SAN_FLAGS) $(SOURCES) -o $@ $(LDLIBS)

build/release/test-kernels: $(SOURCES) $(HEADERS) Makefile
	mkdir -p build/release
	$(CC) $(CPPFLAGS) $(RELEASE_FLAGS) $(SOURCES) -o $@ $(LDLIBS)

test: debug sanitize release
	./build/debug/test-kernels
	./build/sanitize/test-kernels
	./build/release/test-kernels

clean:
	rm -rf build
