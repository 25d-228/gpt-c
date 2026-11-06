CC = xcrun clang
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS)
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2

.PHONY: all debug sanitize release test clean

all: debug

debug: build/debug/dot-demo
sanitize: build/sanitize/dot-demo
release: build/release/dot-demo

build/debug/dot-demo: build/debug/main.o build/debug/dot.o
	$(CC) $(DEBUG_FLAGS) $^ -o $@

build/sanitize/dot-demo: build/sanitize/main.o build/sanitize/dot.o
	$(CC) $(SAN_FLAGS) $^ -o $@

build/release/dot-demo: build/release/main.o build/release/dot.o
	$(CC) $(RELEASE_FLAGS) $^ -o $@

build/debug/%.o: %.c dot.h Makefile
	mkdir -p build/debug
	$(CC) $(DEBUG_FLAGS) -c $< -o $@

build/sanitize/%.o: %.c dot.h Makefile
	mkdir -p build/sanitize
	$(CC) $(SAN_FLAGS) -c $< -o $@

build/release/%.o: %.c dot.h Makefile
	mkdir -p build/release
	$(CC) $(RELEASE_FLAGS) -c $< -o $@

test: debug sanitize release
	./build/debug/dot-demo
	./build/sanitize/dot-demo
	./build/release/dot-demo

clean:
	rm -rf build
