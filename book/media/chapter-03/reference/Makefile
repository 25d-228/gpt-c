CC = xcrun clang
WARNINGS = -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes
BASE_FLAGS = -std=c17 $(WARNINGS) -fno-fast-math -ffp-contract=off
DEBUG_FLAGS = $(BASE_FLAGS) -O0 -g
SAN_FLAGS = $(BASE_FLAGS) -O1 -g -fno-omit-frame-pointer \
            -fsanitize=address,undefined -fno-sanitize-recover=undefined
RELEASE_FLAGS = $(BASE_FLAGS) -O2
LDLIBS = -lm

.PHONY: all debug sanitize release test clean

all: debug

debug: build/debug/numerics
sanitize: build/sanitize/numerics
release: build/release/numerics

build/debug/numerics: numerics.c Makefile
	mkdir -p build/debug
	$(CC) $(DEBUG_FLAGS) numerics.c -o $@ $(LDLIBS)

build/sanitize/numerics: numerics.c Makefile
	mkdir -p build/sanitize
	$(CC) $(SAN_FLAGS) numerics.c -o $@ $(LDLIBS)

build/release/numerics: numerics.c Makefile
	mkdir -p build/release
	$(CC) $(RELEASE_FLAGS) numerics.c -o $@ $(LDLIBS)

test: debug sanitize release
	./build/debug/numerics
	./build/sanitize/numerics
	./build/release/numerics

clean:
	rm -rf build
