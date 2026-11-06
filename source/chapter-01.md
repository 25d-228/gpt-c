# Chapter 1 — Native C builds and a debuggable numerical program

A numerical kernel is a function that performs a small piece of arithmetic on memory supplied by its caller. Before building a decoder, we need to know what the compiler can check about that function, what its caller must guarantee, and how to inspect an incorrect result. A three-element dot product is enough to make those boundaries visible.

By the end of this chapter, one header and two C source files form a complete executable. It has separate debug, sanitizer, and optimized builds. You can stop inside its accumulation loop, explain every memory read, and distinguish a bad declaration, a missing implementation, and an invalid runtime access. No tensor library, Python interpreter, model weights, or later course files are needed.

> **Learning objectives**
> Build and link separate C translation units; describe a borrowed pointer-and-length interface; trace a reduction over contiguous arrays; inspect its state in LLDB; and diagnose an isolated out-of-bounds read without putting undefined behavior in the normal tests.

**Starting point:** basic C expressions, loops, and functions, with no existing application code. **Implementation estimate: 1.75 hours**—0.50 for setup and builds, 0.75 for the program, and 0.50 for debugger work and checks. Reading is separate. Record your actual hands-on time rather than stretching a finished exercise to match the estimate.

## Establish the native build environment

In Terminal, inspect the tools already available:

```sh title="Terminal — observation only"
sw_vers
uname -m
xcode-select --print-path
xcrun --find clang
xcrun clang --version
xcrun lldb --version
make --version
```

For native work on the M4, `uname -m` should report `arm64`. We will also inspect the executable, because the shell's architecture and a compiler's selected target are different observations. Do not copy a particular macOS version from an example into your setup requirements.

`xcrun` locates a tool in the selected Apple developer environment. A working Command Line Tools installation is sufficient for this example; the full Xcode application is not required. When the tools are missing, Apple's installer can be requested with:

```sh title="Terminal — only when Command Line Tools are missing"
xcode-select --install
```

After installation, repeat the observation commands. An invalid active developer directory is a tool-selection problem, not a reason to delete unrelated installations. Apple's [command-line tools installation page][apple-install] and [TN2339 sections on installation and tool selection][apple-faq] describe these mechanisms. The latter is an archived reference: its example operating-system versions are not requirements for this course.

We use **C17 mode**, a working C standard library, debug information, and address/undefined-behavior sanitizer support. The small builds below exercise those features. There is no exact compiler-version gate. A compatible alternative compiler is acceptable; the Makefile provides an explicit override.

Create the example directory inside your learner workspace, not inside the book's generated website:

```sh title="Terminal — from the learner workspace root"
mkdir -p examples/01-build
cd examples/01-build
```

All subsequent shell commands in this chapter run from `examples/01-build/` unless stated otherwise. Write these four persistent files:

```text
examples/01-build/
    dot.h
    dot.c
    main.c
    Makefile
```

The chapter contains their complete contents. Companion source files—[dot.h](https://25d-228.github.io/gpt-c/media/chapter-01/reference/dot.h), [dot.c](https://25d-228.github.io/gpt-c/media/chapter-01/reference/dot.c), [main.c](https://25d-228.github.io/gpt-c/media/chapter-01/reference/main.c), and [Makefile](https://25d-228.github.io/gpt-c/media/chapter-01/reference/Makefile)—contain the same program. Write your working copy in the example directory above. The later `build/` directory contains disposable outputs and the two temporary fault exercises. Do not store your own persistent source there.

## Start with a result that can be calculated by hand

Let $a$ and $b$ each have $n$ coordinates. Their dot product is the scalar

$$
s = \sum_{i=0}^{n-1} a_i b_i.
$$

Here $n$ is the number of elements, $i$ is a zero-based coordinate, $a_i$ and $b_i$ are the values at that coordinate, and $s$ is one number rather than another vector. The summation sign means “multiply corresponding coordinates and add the products.” This is a **reduction**: several input values become one accumulated result.

For $a=[1,2,3]$ and $b=[4,5,6]$,

$$
s = 1\cdot4 + 2\cdot5 + 3\cdot6 = 4+10+18 = 32.
$$

The input shapes are `[3]` and `[3]`; the output is a scalar. In C, each input is a contiguous array of `float`. Element `i` is `i * sizeof(float)` bytes after the first element, but C's subscript operator performs the scaling for us. On the target, a `float` occupies four bytes; the program prints `sizeof(float)` rather than baking that number into address arithmetic.

| Coordinate `i` | Offset within each input on the target | `a[i]` | `b[i]` | Product | Running sum after addition |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 bytes | 1 | 4 | 4 | 4 |
| 1 | 4 bytes | 2 | 5 | 10 | 14 |
| 2 | 8 bytes | 3 | 6 | 18 | 32 |

At `i == 3`, the loop must stop. An array of three elements has indices `0`, `1`, and `2`; its length is not a valid index. The two arrays are independently contiguous. There is no guarantee that one immediately follows the other in memory.

Later, a matrix product will compute many such reductions, with different starting addresses and strides. For now, there is no hidden matrix machinery: the multiply and add happen in our loop.

## Declare the memory contract before writing the loop

A function declaration states types that other source files can use. It does not supply a function body. Create the complete header:

```c title="examples/01-build/dot.h"
#ifndef DL_EXAMPLE_DOT_H
#define DL_EXAMPLE_DOT_H

#include <stddef.h>

/*
 * Read n initialized floats from each input and return their dot product.
 * For n > 0, a and b must each address at least n live float elements.
 * For n == 0, either pointer may be NULL; no input is accessed.
 * Inputs may overlap. This function writes neither input, retains no pointer,
 * and performs no allocation. Invalid extents are a caller error, not a status.
 */
float dot_f32(const float *a, const float *b, size_t n);

#endif
```

The first two lines and the final `#endif` form an **include guard**. On the first inclusion within a translation unit, the preprocessor defines `DL_EXAMPLE_DOT_H` and keeps the declarations. On another inclusion, the name is already defined, so the guarded contents are skipped. This is a source-processing mechanism, not a runtime branch.

`<stddef.h>` declares `size_t`, the unsigned integer type used for object sizes and the result of `sizeof`. Using the same type for `n` and the loop index avoids a signed/unsigned comparison. It does **not** make a length trustworthy or make arbitrary multiplication overflow-safe. Checked size calculations enter in Chapter 2.

Read `const float *a` as “`a` is a pointer through which the function reads floats but does not modify them.” The `*` is part of the pointer declaration. In an expression, `*a` would instead dereference the pointer and designate its first element. `const` here qualifies the pointed-to floats; it does not make the local pointer variable itself unchangeable. A pointer with this type may also refer to otherwise mutable storage. See the GNU C manual's [pointer declarations][c-pointers] and [`const` discussion][c-const].

An input pointer is **borrowed**: the caller provides storage that stays alive during the call. The function neither owns nor frees that storage, and it keeps no pointer after returning. Since it only reads, `a` and `b` may refer to the same array. There is no `restrict` promise here; that keyword will be introduced before an optimization needs it.

Most importantly, an ordinary C pointer carries **no element count**. The callee receives an address and a separate `n`, not a checked slice. Writing a parameter as `const float a[3]` would not automatically enforce three accessible elements; ordinary array parameter declarations are adjusted to pointer types. The GNU C manual explains this adjustment under [function parameters][c-parameters]. We use the pointer form to keep the limitation visible.

The comment adds requirements that the type system does not establish. With positive `n`, both pointers must address enough initialized, live elements, and those elements must not be changed during this call. A non-null check alone would not prove any of that. This teaching kernel has no error-status result: a legitimate dot product can be zero, so zero must not double as “bad pointer.” Its valid calls are defined by the contract above.

The empty case is deliberate. `dot_f32(NULL, NULL, 0)` will return zero without accessing an input. `NULL` represents a null pointer when used as a pointer argument; it does not allocate an empty array. We cannot dereference it. We can nevertheless pass it to a function whose zero-length path performs no dereference or pointer arithmetic.

## Define the reduction in one source module

Create the complete implementation:

```c title="examples/01-build/dot.c"
#include "dot.h"

float dot_f32(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float product = a[i] * b[i];
        sum = sum + product;
    }
    return sum;
}
```

Including its own header lets the compiler compare this definition with the interface seen by callers. Quotes in `#include "dot.h"` select a project header; angle brackets in `#include <stddef.h>` select a standard header through the toolchain's header search. Do not include `dot.c` from another `.c` file. We will compile it separately and link its object file.

`0.0f` is a `float` literal; without the `f`, `0.0` has type `double`. Our accumulator and product are floats, matching the `f32` naming on the target. This is not a claim that standard C requires every implementation's `float` to use the same representation. Chapter 3 will examine representation and rounding explicitly.

The loop initializes `i` to zero, checks `i < n` before each body execution, then increments `i`. `a[i]` designates the element reached by advancing `i` float elements from `a`, not `i` bytes. `product` is a new local value for the iteration. The assignment to `sum` incorporates it into the running total.

We can describe the loop's state without guessing from the final printout. After $k$ iterations, where $0\leq k\leq n$, the accumulator represents the prefix reduction

$$
s_k = \sum_{j=0}^{k-1} a_j b_j,
\qquad s_0=0.
$$

The new symbol $j$ indexes terms already processed; $k$ counts completed iterations. The loop body adds the next product, taking $s_k$ to $s_{k+1}$. At termination, $k=n$. For these small integer-valued inputs the results are exactly representable, so the earlier table describes the actual values. With arbitrary floating-point data, each arithmetic step can round; this formula expresses the intended reduction, not a promise of exact real arithmetic.

For `n == 0`, the first condition `0 < 0` is false. Neither `a[i]` nor `b[i]` is evaluated, and `sum` remains zero. This is why the implementation meets the empty-case contract without special pointer handling. It also avoids computing `n - 1`: subtracting one from an unsigned zero would produce a large unsigned value, not a useful “last index.”

There is no allocation and no cleanup path in this function. The returned scalar is a value, not a pointer to the local `sum`. The caller's arrays remain the caller's responsibility.

## Give the kernel a complete caller and executable checks

The following two blocks, in order, form **one complete `main.c`**. There is no omitted helper. Begin with includes and the normal/empty evaluations:

```c title="examples/01-build/main.c — first part"
#include "dot.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    const size_t n = sizeof a / sizeof a[0];

    const float result = dot_f32(a, b, n);
    const float empty = dot_f32(NULL, NULL, 0);

    printf("dot = %.0f\n", (double)result);
    printf("empty dot = %.0f\n", (double)empty);
    printf("elements = %zu, bytes per element = %zu\n", n, sizeof a[0]);
```

`int main(void)` declares a function with no parameters and an integer exit status. The `(void)` matters: in C17, an empty parameter list in an ordinary declaration does not express the same prototype. `<stdio.h>` supplies declarations for output functions and the standard error stream. `<stdlib.h>` supplies `EXIT_SUCCESS` and `EXIT_FAILURE`, which we use below.

`a` and `b` are actual array objects here. Their storage has automatic duration: it remains alive while this invocation of `main` is executing. Each array occupies space for three elements, and the initializer supplies all three values. In a call such as `dot_f32(a, b, n)`, each array expression is converted to a pointer to its first element. The arrays are not copied into the callee.

`sizeof a` is the size of the entire array, while `sizeof a[0]` is the size of one element. Their ratio is therefore three. This works **where `a` is an array**. Inside `dot_f32`, `sizeof a` would give the size of its pointer parameter, not the caller's array. The [GNU C manual's type-size discussion][c-size] describes this distinction. A pointer to a large tensor and a pointer to one float need not differ in size; the pointed-to extent is separate information.

The `%zu` conversion prints a `size_t` value without assuming that it is an `int` or a particular unsigned integer type. `%.0f` prints a floating value with zero digits after the decimal point. `printf` takes a variable number of arguments; a float supplied in that variable argument list is promoted to `double`. The casts above make the expected argument type explicit; they do not change the kernel's accumulation precision. Standard floating and integer output conversions are described in the [GNU C library reference][c-output].

The display format is **not** a correctness test: a nearby wrong result could still print `32`. Append these checks and the end of `main` immediately after the first block:

```c title="examples/01-build/main.c — second part"
    if (result != 32.0f || empty != 0.0f) {
        fprintf(stderr, "FAIL: normal or empty dot product\n");
        return EXIT_FAILURE;
    }
    if (dot_f32(a, b, 1) != 4.0f) {
        fprintf(stderr, "FAIL: one-element prefix\n");
        return EXIT_FAILURE;
    }
    if (dot_f32(a + 1, b + 1, n - 1) != 28.0f) {
        fprintf(stderr, "FAIL: two-element suffix\n");
        return EXIT_FAILURE;
    }
    if (dot_f32(a, a, n) != 14.0f) {
        fprintf(stderr, "FAIL: overlapping read-only inputs\n");
        return EXIT_FAILURE;
    }

    puts("all checks passed");
    return EXIT_SUCCESS;
}
```

`||` means logical “or”; the first failure branch is taken when either the normal result or the empty result is wrong. `fprintf(stderr, ...)` sends a diagnostic to standard error. Returning `EXIT_FAILURE` makes a failed check visible to the shell and to `make`, not just to a person reading text. `puts` writes the success message with a newline.

The other calls probe the interface. A one-element prefix returns $1\cdot4=4$. Advancing both pointers by one skips the first pair, so the two-element suffix returns $2\cdot5+3\cdot6=28$. Here `n` is known to be three, making `n - 1` safe. In general, adjust the available length whenever you advance a pointer. Reusing length three after `a + 1` would overread the original three-element array.

Passing `a` twice computes $1^2+2^2+3^2=14$. It is valid because both inputs are read-only. No temporary copy or disjointness assumption is needed. These five calls test normal, empty, single-element, offset, and overlapping-input cases.

Exact equality is appropriate **for these fixtures**: the inputs, products, and sums are small integers exactly representable in the target's floating-point format. It is not our general tensor-testing policy. Chapter 3 introduces tolerances and non-finite checks for less accommodating calculations.

We use explicit branches instead of making test execution depend on `assert`, which can be disabled by defining `NDEBUG`. The numerical checks therefore remain active in the optimized executable too. The output calls are human-readable diagnostics; the process exit status is the test result, not a durable output-file protocol.

## Follow the source through compilation and linking

A **translation unit** is one C source file after preprocessing its includes and directives. Our program has two: one from `main.c`, another from `dot.c`. Both see a declaration of `dot_f32` from the header; only the second contains its definition.

Conceptually, the toolchain proceeds through preprocessing, type/syntax checking and code generation, assembly into object files, then linking those objects into an executable. An object file contains compiled code and symbols, but is not yet this complete program. The compiler driver coordinates these stages; we do not invoke the linker directly. These stages and their selection flags are documented in the [Clang command guide][clang-driver].

Before using the commands, read their options:

| Option | Role in this example |
|---|---|
| `-std=c17` | Select C17 source-language rules. |
| `-Wall -Wextra -Wpedantic` | Enable useful warning groups and language-extension diagnostics; this is not a proof of correctness. |
| `-O0` | Disable most optimization for straightforward source-level stepping. |
| `-g` | Emit debugging information used to associate instructions with source and variables. |
| `-c` | Produce an object file without linking an executable. |
| `-o path` | Name the output. |
| `-E` | Stop after preprocessing. |
| `-S` | Produce assembly text rather than an object or linked executable. |

Compile and link explicitly once, so the later Makefile is not mysterious:

```sh title="Terminal — separate compilation, then linking"
mkdir -p build/manual
xcrun clang -std=c17 -Wall -Wextra -Wpedantic -O0 -g \
    -c main.c -o build/manual/main.o
xcrun clang -std=c17 -Wall -Wextra -Wpedantic -O0 -g \
    -c dot.c -o build/manual/dot.o
xcrun clang -g build/manual/main.o build/manual/dot.o \
    -o build/manual/dot-demo
./build/manual/dot-demo
```

The trailing backslash on a shell line continues that command onto the next line. Keep it as the last character, without spaces after it. `mkdir -p` creates the output directory when necessary. The `./` execution prefix selects the program at that relative path rather than searching the shell's command path.

Expected program output on the target:

```text
dot = 32
empty dot = 0
elements = 3, bytes per element = 4
all checks passed
```

The compiler may emit its own diagnostics before the run. There should be none for these valid listings with the shown options. A nonzero program exit is a failed check even when some earlier lines look right.

You can also inspect two intermediate forms:

```sh title="Terminal — inspect the build stages"
xcrun clang -std=c17 -E dot.c -o build/manual/dot.i
xcrun clang -std=c17 -O0 -S dot.c -o build/manual/dot.s
```

Open `dot.i` in an editor and find the declaration and definition of `dot_f32`. Header contents and preprocessing markers appear around them. Open `dot.s` to see that a C function becomes target-specific instructions; you do not need to learn the assembly language here. Neither command creates a runnable program.

The declaration is enough to compile a call. The definition must still be available at link time. The header itself is not an object file and should not be passed to the link command. Include guards prevent repeated header processing within a translation unit; they do not prevent duplicate function definitions across separate translation units.

## Make builds repeatable without mixing configurations

A debugger build favors inspectable state. A sanitizer build adds runtime checks. An optimized build changes code generation and should not inherit sanitizer instrumentation accidentally. Put their object files and executables in separate directories.

Our additional warning options are `-Wconversion`, for potentially value-changing implicit conversions; `-Wshadow`, for declarations that hide another name; and `-Wstrict-prototypes`, for function declarations that omit parameter types. The [Clang diagnostics reference][clang-warnings] documents their categories. Warnings are deliberately not all promoted to errors: compiler releases can add diagnostics. Investigate new warnings instead of suppressing them merely to copy a version's output.

The sanitizer configuration uses `-fsanitize=address,undefined` during **both compilation and linking**. AddressSanitizer checks instrumented memory accesses. UndefinedBehaviorSanitizer checks selected operations such as signed integer overflow and null/misaligned accesses. `-fno-sanitize-recover=undefined` makes a detected undefined-behavior check terminate instead of reporting and continuing. `-fno-omit-frame-pointer` helps make call stacks usable. The [AddressSanitizer usage guide][asan] and [UndefinedBehaviorSanitizer usage guide][ubsan] explain these flags and their limits.

We use `-O1` for the sanitizer build and `-O2` for the optimized build. None uses `-ffast-math`, whose assumptions can alter floating-point behavior and reasoning about non-finite values; see [Clang's floating-point guide][clang-fp]. A named `product` in source is useful for debugging, not a promise of a particular number of machine instructions or universal bitwise agreement.

A Makefile rule has a **target**, its **prerequisites**, and indented shell commands called its **recipe**. A file target is rebuilt when missing or older than a prerequisite. The recipe lines below begin with a **literal tab**, not a run of spaces. The GNU make manual describes [rule syntax][make-rules] and [automatic variables][make-auto].

Create this complete Makefile:

```makefile title="examples/01-build/Makefile"
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
```

`$(NAME)` expands a Makefile variable. The `=` assignments above define text expanded when used. `CC` defaults to `xcrun clang` in this file; a command-line assignment such as `make CC=clang test` overrides it. That override is useful with a compatible non-Apple compiler, including a portable check on another operating system. It does not simulate an M4 run.

In a recipe, `$@` is the target filename, `$<` the first prerequisite, and `$^` the list of prerequisites without duplicates. For the debug link rule, `$^` is exactly the two object filenames. In the pattern rule, `%` matches the shared stem: `build/debug/main.o` uses `main.c`. Naming `dot.h` and `Makefile` as prerequisites ensures that editing the interface or the recorded flags rebuilds the affected objects. This small program needs no automatic dependency generator.

`all` is the first ordinary target and requests `debug` by default. `.PHONY` marks names such as `test` and `clean` as commands to perform rather than outputs to consider current merely because a same-named file exists. `clean` removes only the example's `build/` tree. Run it from this directory and never place irreplaceable work in that tree. Phony targets and command-line overrides are described in the [GNU make manual][make-manual].

Now build and check all three configurations:

```sh title="Terminal — normal builds and checks"
make test
file build/debug/dot-demo
```

`make` echoes commands; each executable then prints the four output lines already shown. A failing check returns nonzero and stops the normal recipe. The `file` output for the native target should identify an `arm64` Mach-O executable. An `x86_64` result calls for checking the compiler target and how the terminal/toolchain was launched, rather than interpreting that run as native M4 behavior.

A second `make test` normally skips compilation but runs the tests again. Changing `dot.c` rebuilds its object in each configuration, then relinks. Changing `dot.h` rebuilds both objects in each configuration. Changing the compiler through `CC=...` is **not** a file timestamp change; after switching compilers, use:

```sh title="Terminal — rebuild after a toolchain change"
make clean
make CC=clang test
```

On the standard Apple path, omit `CC=clang` to keep the selected `xcrun` compiler. `make -n test` prints the planned commands without executing them. Never time a sanitizer executable and describe the result as optimized-kernel performance; this chapter does not yet perform a useful benchmark.

## Stop at a particular accumulation in LLDB

Use the unoptimized debug executable, not the release executable:

```sh title="Terminal — launch the debugger"
make debug
xcrun lldb ./build/debug/dot-demo
```

LLDB's prompt accepts debugger commands, not shell commands. A **breakpoint** pauses execution at a selected function or source location. A **stack frame** describes one active function call, including its arguments and local state. We will first stop at the function, then at its second accumulation. The [LLDB tutorial][lldb] documents breakpoints, execution control, and `frame variable`.

Enter this first group at the LLDB prompt:

```text title="LLDB — enter the reduction"
breakpoint set --name dot_f32
run
frame variable n
frame variable 'a[0]' 'b[0]'
thread backtrace
```

On the first call, expect `n` to be three, `a[0]` to be one, and `b[0]` to be four. Type-name spelling, addresses, and frame numbers can differ by toolchain. The backtrace should include `dot_f32` called by `main`. At function entry, do not assume that every local has already been initialized; the source cursor indicates the next statement associated with the stop.

In the exact `dot.c` listing above, line 8 is `sum = sum + product;`. Set a second breakpoint there, conditional on `i == 1`:

```text title="LLDB — stop before the second addition"
breakpoint set --file dot.c --line 8 --condition 'i == 1'
breakpoint disable 1
continue
frame variable i sum product
next
frame variable sum
```

These breakpoint numbers assume a fresh session and the commands shown. Use the numbers LLDB reports when they differ. After editing the source, find the accumulation line with `source list --name dot_f32` and replace `8` with its current line number.

The condition uses zero-based indexing: `i == 1` is the second element. Immediately **before** executing the addition, expect the state below. After `next` steps over that statement, `sum` should be 14.

| Observation | Expected state |
|---|---|
| Before the selected addition | `i = 1`, `sum = 4`, `product = 10` |
| After `next` | `sum = 14` |

This directly tests the earlier prefix invariant. A final value of 32 alone would not reveal whether a wrong intermediate had later been compensated by another mistake.

Disable the conditional breakpoint so it does not stop later test calls, then finish:

```text title="LLDB — finish the checks"
breakpoint disable 2
continue
quit
```

The executable should finish its normal checks and exit successfully. With an optimized binary, a local might be unavailable or source stepping might appear to jump; rebuild with the debug target rather than concluding that LLDB changed the mathematics. On macOS, if source line information is missing despite `-g`, keep the original `.o` files alongside the executable while debugging. `xcrun dsymutil build/debug/dot-demo` can collect their debug information into a `.dSYM` bundle; see the [dsymutil command guide][dsymutil] and [LLDB's symbols-on-macOS documentation][lldb-symbols]. Do not remove the objects before that collection.

A debugger launch-permission failure is a debugger/environment problem, distinct from a failed numerical check. Do not disable system security broadly to make an example launch. The program remains testable directly while you inspect the local tool-selection or permission message.

## Separate a compiler warning from a linker failure

The compiler can identify a mismatch that is visible in one translation unit. It cannot conjure an implementation absent from the link inputs. We will provoke each diagnostic in a disposable directory without altering the working kernel.

First create the directory:

```sh title="Terminal — isolated diagnostics"
mkdir -p build/diagnostics
```

Create this complete temporary source file. Its format is intentionally wrong; **do not run it**:

```c title="examples/01-build/build/diagnostics/format.c"
#include <stddef.h>
#include <stdio.h>

int main(void)
{
    const size_t n = 3;
    printf("elements = %d\n", n);
    return 0;
}
```

Use `-fsyntax-only` to ask for preprocessing and language checks, without producing an executable:

```sh title="Terminal — diagnose but do not execute the bad format"
xcrun clang -std=c17 -Wall -Wextra -Wpedantic -fsyntax-only \
    build/diagnostics/format.c
```

Expect a diagnostic that `%d` expects an `int`, while the supplied value is a `size_t` (possibly displayed as its underlying unsigned type). A warning need not make the compiler return nonzero. The mismatch is not repaired by the fact that the value happens to be three; the format specifies how the variadic argument must be interpreted. Replace `%d` with `%zu` in this temporary file and repeat the check. It should now be quiet.

Next omit the kernel's object from a link:

```sh title="Terminal — intentionally incomplete link"
xcrun clang -g build/debug/main.o -o build/diagnostics/missing-dot
```

Expect a nonzero linker failure naming an unresolved `dot_f32` symbol, possibly with a platform prefix such as `_dot_f32`. `main.o` compiled successfully because its translation unit saw the declaration. The command did not provide the object defining that function. The repair is to link `build/debug/dot.o` as the Makefile already does—not to put the body in the header or suppress the linker error.

A declaration and definition that disagree across files can also be dangerous: ordinary linking is not a general type checker for C interfaces. Including the same header in the implementation and all callers makes such disagreements visible to compilation. This is why `dot.c` includes `dot.h` even though its function definition appears self-contained.

## Let a sanitizer catch a caller that lies about length

A correct loop cannot infer that a caller supplied too large an `n`. We will keep `dot.c` unchanged and create a separate, deliberately invalid caller. This exercise uses AddressSanitizer alone so the intended report is clearly a memory-access report; the normal Makefile still checks with both sanitizers.

Create its disposable directory:

```sh title="Terminal — prepare the fault exercise"
mkdir -p build/fault
```

Then create this complete temporary file:

```c title="examples/01-build/build/fault/overread.c"
#include "dot.h"

#include <stdio.h>

int main(void)
{
    const float a[3] = {1.0f, 2.0f, 3.0f};
    const float b[3] = {4.0f, 5.0f, 6.0f};
    const float result = dot_f32(a, b, 4);
    printf("invalid result = %.0f\n", (double)result);
    return 0;
}
```

The only incorrect decision is the length four for arrays of length three. The fourth loop iteration tries to read element three. Forming a pointer just past an array can be valid; reading an element through that one-past pointer is not. This access has **undefined behavior**: the language makes no reliable promise of a crash, a particular value, or apparent success. An unsanitized run that prints something would not validate it.

`-I.` adds the current directory to the header-search path, letting a source in `build/fault/` include the project header. Compile and link with sanitizer flags at every stage:

```sh title="Terminal — instrument the kernel and faulty caller"
xcrun clang -std=c17 -Wall -Wextra -Wpedantic -O1 -g \
    -fsanitize=address -fno-omit-frame-pointer \
    -c dot.c -o build/fault/dot.o
xcrun clang -std=c17 -Wall -Wextra -Wpedantic -O1 -g -I. \
    -fsanitize=address -fno-omit-frame-pointer \
    -c build/fault/overread.c -o build/fault/overread.o
xcrun clang -g -fsanitize=address \
    build/fault/overread.o build/fault/dot.o -o build/fault/overread

if ./build/fault/overread; then
    printf '%s\n' 'Unexpected normal return; inspect the sanitizer build.'
else
    printf '%s\n' 'Inspect the report above; a nonzero exit alone is not proof.'
fi
```

The shell `if` selects its first branch only when the command exits successfully. The exercise is expected to take the second branch, with an AddressSanitizer report identifying a **stack-buffer-overflow** and a read. It is called a stack-buffer error because the caller's local arrays are stored in its stack frame in this build; no heap allocation was involved.

Do not match the report against one address, process ID, or exact line-number transcript. Read the operation, its size, the first relevant source frame, and the named object whose extent was crossed. On this target the invalid element read is four bytes. The access is in `dot_f32`; the incorrect length originates in `main`. Fixing the fault therefore requires tracing the call, not merely blaming the line that performs the load.

For this case, `a` has room for three four-byte elements: 12 bytes. After reading at offsets 0, 4, and 8, the next iteration requests a read starting at offset 12, outside the array. A report may name either input depending on code generation and access order. The important evidence is an instrumented read beyond one of those three-element arrays.

AddressSanitizer instruments generated code and checks accesses against runtime memory metadata. It reports the first detected problem, rather than providing a usable numerical result from this invalid computation. If source names are absent, consult its [symbolization instructions][asan-symbols]; on macOS, generating a `.dSYM` with `dsymutil` can be necessary. Do not treat an unrelated loader or tool-launch failure as a successful sanitizer test.

Sanitizers also cannot establish the **logical** extent of every numerical view. Suppose the arrays each really contain four initialized floats, but the algorithm is intended to use only their first three. Passing four would be memory-accessible yet mathematically wrong. A hand-calculated expected result or a shape contract must catch that. Likewise, passing the wrong valid row of a matrix can be completely invisible to memory diagnostics.

Restore the ordinary working state after examining the fault:

```sh title="Terminal — remove the temporary exercises and recheck"
make clean
make test
```

`make clean` removes the fault sources and outputs because they were created under `build/`. The four persistent files never changed. The Makefile's `test` target never depends on an intentional fault and should again complete normally.

## Make one small change, then predict the effect

Before running, predict how the loop and its caller should change in these two variations. They need no new infrastructure.

**Change the supplied prefix length.** In a temporary edit to the normal evaluation, use `dot_f32(a, b, n - 1)` instead of length `n`. The result becomes 14, not 32, because the last pair is no longer processed. The existing normal-result check should fail. The sanitizer should remain quiet: accessing a valid prefix is a numerical mismatch here, not a memory violation. Restore the original call afterward.

**Negate one input coordinate.** Change `b[1]` to `-5.0f`. The complete dot product becomes $4-10+18=12$, the suffix becomes $-10+18=8$, the one-element prefix stays 4, and the self-dot of `a` remains 14. Update just those two affected expectations, run all three builds, and confirm the new trace in the debugger. Then restore the original fixture and checks so the chapter leaves the named prerequisite state intact.

These are short coding exercises, not new projects. Their purpose is to separate “the code accesses valid memory” from “the code computes the intended reduction,” and to make a prediction before the program tells you its answer.

## Diagnose the smallest failing boundary

| Symptom | First useful interpretation and action |
|---|---|
| `xcrun` cannot locate the tool | Inspect the selected developer directory and installation; the C source has not yet been tested. |
| `dot.h` is not found | Check the working directory, filename case, and `-I.` for the temporary caller. Do not copy a second header with a diverging declaration. |
| `make` reports “missing separator” | Check that recipe indentation is a tab and that the Makefile was not saved with Markdown fences. |
| Compilation warns about an argument type | Fix the declaration or actual conversion; do not assume a successful link repairs it. |
| Linking reports missing `dot_f32` | Check that the matching `dot.o` is an input to this configuration's link command. |
| Linking reports sanitizer runtime symbols | Compile and link through the same compiler driver with the sanitizer flags; do not reuse an uninstrumented or mismatched configuration casually. |
| LLDB cannot show `sum` | Check the selected binary, `-O0 -g`, source location, and retained debug information; at function entry the assignment may not have executed. |
| A memory report points into the reduction | Inspect the caller's length and the remaining extent after any pointer offset, as well as the loop bound. |
| The checks fail without a sanitizer report | Trace products and accumulation; wrong but in-bounds arithmetic is still wrong. |
| An edit appears to have no effect | Check the executable path and Makefile prerequisites; clean after changing compilers or unrecorded command-line flags. |

## The code state carried into Chapter 2

The persistent result is `dot.h`, `dot.c`, `main.c`, and the Makefile under `examples/01-build/`. Their normal state computes 32, returns zero for the empty call, and passes all five checks in each build configuration. The directory remains independently runnable; it does not depend on future decoder infrastructure.

You have an interface with explicit extent and lifetime assumptions, but the inputs still use fixed local storage. Chapter 2 will build a separate foundation program that acquires storage dynamically, borrows a view into it, rejects impossible sizes, and cleans up correctly after failure. The useful question carried forward is not merely “what address does this pointer contain?” but “who keeps that storage alive, how much is accessible, and what happens when construction fails?”

## Technical references

These references document the language and tool behavior used above. The complete runnable example is in this chapter; none of the links supplies missing program logic.

**Apple tools:** [Installing the command-line tools][apple-install]; [Technical Note TN2339][apple-faq], particularly installing and selecting developer tools. Historical paths and OS examples in TN2339 are not version requirements.

**Compilation and diagnostics:** [Clang command guide][clang-driver], “Description” and “Stage Selection Options”; [diagnostic flag reference][clang-warnings] for the named warning groups; [floating-point behavior][clang-fp] for `-ffast-math` assumptions.

**C interfaces:** GNU C manual, [pointer declarations][c-pointers], [`const`][c-const], [function parameter variables][c-parameters], and [type size][c-size]. These particular language rules are used in C17 mode; the manual also contains GNU extensions that this example does not require. The archived [GNU C library output reference][c-output] documents the standard `%zu` and floating-output conventions used here, not an Apple-library dependency.

**Build descriptions:** GNU make, [rule syntax][make-rules], [automatic variables][make-auto], and [the manual's phony-target and override sections][make-manual]. The Makefile uses established features rather than requiring a recently introduced recipe syntax.

**Runtime inspection:** [LLDB tutorial][lldb], “Setting Breakpoints,” “Controlling Your Program,” and “Examining Stack Frame State”; [Symbols on macOS][lldb-symbols]; [dsymutil description][dsymutil]; [AddressSanitizer usage][asan] and [report symbolization][asan-symbols]; [UndefinedBehaviorSanitizer usage][ubsan].

[apple-install]: https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/
[apple-faq]: https://developer.apple.com/library/archive/technotes/tn2339/_index.html
[clang-driver]: https://clang.llvm.org/docs/CommandGuide/clang.html
[clang-warnings]: https://clang.llvm.org/docs/DiagnosticsReference.html
[clang-fp]: https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior
[c-pointers]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Pointer-Declarations.html
[c-const]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/const.html
[c-parameters]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Function-Parameter-Variables.html
[c-size]: https://www.gnu.org/software/c-intro-and-ref/manual/c-intro-and-ref.html#Type-Size
[c-output]: https://ftp.gnu.org/old-gnu/Manuals/glibc-2.2.3/html_chapter/libc_12.html
[make-rules]: https://www.gnu.org/software/make/manual/html_node/Rule-Syntax.html
[make-auto]: https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
[make-manual]: https://www.gnu.org/software/make/manual/make.html
[lldb]: https://lldb.llvm.org/use/tutorial.html
[lldb-symbols]: https://lldb.llvm.org/use/symbols.html
[dsymutil]: https://llvm.org/docs/CommandGuide/dsymutil.html
[asan]: https://clang.llvm.org/docs/AddressSanitizer.html#usage
[asan-symbols]: https://clang.llvm.org/docs/AddressSanitizer.html#symbolizing-the-reports
[ubsan]: https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html#usage
