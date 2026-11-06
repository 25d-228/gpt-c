CC = xcrun clang
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS)
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2

.PHONY: all debug sanitize release test clean

all: debug

debug: build/debug/storage-demo
sanitize: build/sanitize/storage-demo
release: build/release/storage-demo

build/debug/storage-demo: build/debug/main.o build/debug/buffer.o
	$(CC) $(DEBUG_FLAGS) $^ -o $@

build/sanitize/storage-demo: build/sanitize/main.o build/sanitize/buffer.o
	$(CC) $(SAN_FLAGS) $^ -o $@

build/release/storage-demo: build/release/main.o build/release/buffer.o
	$(CC) $(RELEASE_FLAGS) $^ -o $@

build/debug/%.o: %.c buffer.h Makefile
	mkdir -p build/debug
	$(CC) $(DEBUG_FLAGS) -c $< -o $@

build/sanitize/%.o: %.c buffer.h Makefile
	mkdir -p build/sanitize
	$(CC) $(SAN_FLAGS) -c $< -o $@

build/release/%.o: %.c buffer.h Makefile
	mkdir -p build/release
	$(CC) $(RELEASE_FLAGS) -c $< -o $@

test: debug sanitize release
	./build/debug/storage-demo
	./build/sanitize/storage-demo
	./build/release/storage-demo

clean:
	rm -rf build
