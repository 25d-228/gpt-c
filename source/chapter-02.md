# Chapter 2 — Owned buffers, borrowed views, and failure cleanup

The dot product in Chapter 1 borrowed two arrays that already existed. A decoder must also create arrays whose sizes are known only while the program runs, give different operators access to parts of them, and release every allocation when a later step fails. A correct arithmetic loop is not enough if its input has already been freed or its byte count wrapped around before allocation.

This chapter builds a separate storage example. One owner holds six floats representing a two-by-three matrix. A row view and a column view describe different traversals of those same floats without copying them. A second routine acquires two buffers, deliberately fails at either allocation, and leaves its caller's result unchanged. The normal executable never reads freed memory; a separately built diagnostic shows what that error looks like.

> **Learning objectives**
> Represent unique ownership and non-owning strided views; derive overflow-safe size and range checks; initialize every resource before a cleanup path can reach it; inject allocation failures without exhausting memory; and distinguish `const`, aliasing, and the specific promise made by `restrict`.

**Prerequisite:** [Chapter 1 — Native C builds and a debuggable numerical program](https://25d-228.github.io/gpt-c/chapter-01.html), including its pointer-and-length contract and three build configurations. Its files remain unchanged. This example uses no future tensor code and no model assets.

**Implementation estimate: 2.50 hours**—1.50 for the storage module and driver, and 1.00 for boundary cases, failure injection, and sanitizer diagnosis. Reading is separate. Record actual hands-on time rather than adding work merely to fill the estimate.

## Put six values in one allocation

Let $R$ denote the number of rows and $C$ the number of columns. We will use $R=2$ and $C=3$ and store the matrix in **row-major order**: finish one row before starting the next.

$$
A = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},
\qquad N=RC=6.
$$

Here $N$ is the number of float elements, not a byte count. For zero-based coordinates $r$ and $c$, with $0\leq r<R$ and $0\leq c<C$, the flat element offset is

$$
\operatorname{offset}(r,c)=rC+c.
$$

There are $r$ complete rows before row $r$, each with $C$ elements, followed by $c$ elements within the selected row. The allocation is just `[1, 2, 3, 4, 5, 6]`; it contains neither row objects nor pointers to rows. A `float **` would describe a different representation and is not needed here.

| Coordinate | Flat offset | Value | Byte offset when `sizeof(float) == 4` |
|---|---:|---:|---:|
| `(0, 0)` | 0 | 1 | 0 |
| `(0, 1)` | 1 | 2 | 4 |
| `(0, 2)` | 2 | 3 | 8 |
| `(1, 0)` | 3 | 4 | 12 |
| `(1, 1)` | 4 | 5 | 16 |
| `(1, 2)` | 5 | 6 | 20 |

Row 1 visits offsets `3, 4, 5` and sums to `15`. Column 1 visits offsets `1, 4` and sums to `7`. A **stride** is the distance, measured here in float elements, between consecutive values of a view: one for a row, three for a column. Pointer addition already scales by the pointed-to type; never multiply a float-pointer offset by `sizeof(float)` a second time. See the GNU C manual's [array indexing explanation][c-arrays].

Create the new directory from the learner workspace root:

```sh title="Terminal — from the learner workspace root"
mkdir -p examples/02-storage
cd examples/02-storage
```

The four persistent files are `buffer.h`, `buffer.c`, `main.c`, and `Makefile`. For each filename, append its code blocks in the order shown. Blocks labeled as an illustration or a fault exercise are not part of those four files. The companion copies are also available as [buffer.h][source-header], [buffer.c][source-buffer], [main.c][source-main], and [Makefile][source-make]; the complete implementations are explained below.

## Separate an owner from a description of its elements

A `struct` groups named fields into one value. `typedef` gives a type a short name; it does not allocate anything or add runtime behavior. An `enum` supplies named integer constants. We use a status value rather than returning a float sentinel: zero is a valid sum, and a null data pointer is the intended representation of an empty buffer.

Begin the header with the status and two data types:

```c title="examples/02-storage/buffer.h — status and storage types"
#ifndef DL_EXAMPLE_BUFFER_H
#define DL_EXAMPLE_BUFFER_H

#include <stddef.h>

typedef enum {
    STORAGE_OK = 0,
    STORAGE_ARGUMENT,
    STORAGE_STATE,
    STORAGE_OVERFLOW,
    STORAGE_ALLOC,
    STORAGE_RANGE
} StorageStatus;

typedef struct {
    float *data;
    size_t count;
} FloatBuffer;

typedef struct {
    const float *data;
    size_t count;
    size_t stride;
} FloatView;

```

`FloatBuffer` is an **owner by convention**. A live owner contains the original allocation pointer and its count of float elements. Exactly one owner is responsible for calling `free`. C does not enforce that rule for us. The permitted states are deliberately small:

| State | `data` | `count` | What is valid? |
|---|---|---:|---|
| Empty | `NULL` | 0 | Construct, destroy, or borrow an empty view |
| Live | Original allocation pointer | Positive | Initialize/use elements, borrow views, then destroy |
| Invalid | Any inconsistent or invented combination | Any | Not a legitimate owner; do not pass it to destruction |

A local declaration such as `FloatBuffer matrix = {.data = NULL, .count = 0};` uses **designated initializers**: each `.field` names the member being initialized. This initializes field values; it is not an assertion that every representation byte of the struct is zero. The [GNU C manual's initializer discussion][c-designated] covers the syntax. A bare `FloatBuffer matrix;` has no such initialization and is not safe to hand to cleanup.

The owner descriptor and its allocation have different lifetimes. A local `FloatBuffer` itself has automatic storage duration. The allocation it points to remains allocated until explicitly freed, even if the descriptor goes out of scope. Losing the only owner without freeing its allocation is a leak; leaving a block does not call a C destructor.

`FloatView` describes a traversal. Its `data` points to the first selected element, `count` says how many elements are selected, and `stride` says how far apart they are. The `const float *` prevents writes **through that pointer**. It neither freezes the allocation nor makes it live longer. A write through the owner will be visible through every view that selects that element.

For a struct value, `matrix.data` selects a field. For a pointer to a struct, `owner->data` means `(*owner).data`. A `const FloatBuffer *owner` prevents changing the descriptor through `owner`; it does not recursively turn its `float *data` field into a pointer to immutable floats. The view therefore has its own `const float *` field, and our view-building function performs no element writes.

## Make allocation failure a controllable input

A real allocator can fail, but requesting enough memory to exhaust the machine is a poor test. Instead, the constructor will accept a function that either delegates to `malloc` or returns `NULL` at a selected call. This is a small test seam, not a general-purpose allocator framework.

`AllocateFn` below is a **function-pointer type**. Read `void *(*AllocateFn)(size_t bytes, void *context)` from the name outward: it is a pointer to a function taking a byte count and an object pointer, returning an object pointer. The parentheses around `*AllocateFn` distinguish a pointer to a function from a function returning a pointer. The [function-pointer declaration reference][c-function-pointer] explains this syntax.

The `context` lets a callback receive its own state without a global variable. It has type `void *`, a generic object pointer. A callback that knows the real type converts it back to that object-pointer type before using it. We do not convert a function pointer to `void *`; the callback and its context are separate arguments.

Append the remaining header:

```c title="examples/02-storage/buffer.h — contracts and declarations"
/* Return fresh malloc-compatible storage, or NULL. No output mutation.
 * The context is borrowed only during the call; no pointer is retained.
 * Successful results must be releasable by the standard free function. */
typedef void *(*AllocateFn)(size_t bytes, void *context);

const char *storage_status_name(StorageStatus status);
StorageStatus size_mul(size_t a, size_t b, size_t *out);
void *system_allocate(size_t bytes, void *context);

/* out must point to an initialized, empty owner: {NULL, 0}.
 * On success, positive count owns count UNINITIALIZED floats.
 * Zero count succeeds without allocating. allocate must still be non-null.
 * On reported failure, out is unchanged. Never copy a live owner. */
StorageStatus buffer_create(FloatBuffer *out, size_t count,
                            AllocateFn allocate, void *context);

/* Accept NULL, an empty owner, or the unique live owner. Reset it to empty.
 * All views of a freed allocation become invalid. Does not clear copies. */
void buffer_destroy(FloatBuffer *buffer);

/* owner must be an initialized empty/live owner, with truthful extent.
 * offset/count/stride are in FLOAT ELEMENTS; stride must be positive.
 * Empty requests allow offset <= owner->count and produce {NULL, 0, 1}.
 * On reported failure, out is unchanged. No allocation or lifetime extension. */
StorageStatus buffer_view(const FloatBuffer *owner, size_t offset,
                          size_t count, size_t stride, FloatView *out);

/* view must describe live initialized elements, normally via buffer_view.
 * Reads only, allocates nothing, retains nothing. Empty sum is zero.
 * Forged metadata or a destroyed owner violates the caller contract. */
float view_sum(FloatView view);

#endif
```

A successful callback must return **fresh storage compatible with standard `free`**, sufficiently sized and aligned for the requested floats. It may not hand back a stack array, part of an arena requiring a different deallocator, or a pointer already owned elsewhere. Because every successful result follows the same release rule, `FloatBuffer` need not retain a callback, context, or custom deallocator. Neither constructor nor owner retains the context after the synchronous call. The callback must not indirectly modify the constructor's output descriptor.

That restriction is intentional. Adding arbitrary allocation families would require storing and preserving the matching deallocation operation and its context lifetime. Nothing in this example claims to support them.

All output pointers must designate live writable objects of the documented type. A null check can reject `NULL`; it cannot prove that a non-null pointer has the required extent or lifetime. The same limitation applies to the owner's count. Public fields make the representation visible for learning, but inventing a count does not enlarge an allocation.

## Check the integer calculation before asking for memory

The float count and byte count are separate calculations. For a general matrix we first check $R\times C$, then check $N\times\operatorname{sizeof}(\texttt{float})$. A valid first product does not imply that the second fits.

Let $M=\texttt{SIZE\_MAX}$, the largest `size_t` value. For nonnegative integers $a$ and $b$, when $a>0$ the product fits exactly when

$$
ab\leq M
\quad\Longleftrightarrow\quad
b\leq\left\lfloor\frac{M}{a}\right\rfloor.
$$

We can test the division inequality without evaluating the potentially overflowing multiplication. When $a=0$, the product is zero and division by $a$ must be skipped.

This matters because `size_t` arithmetic is unsigned. A product outside its range wraps; assigning the wrapped result to a larger allocation variable afterward cannot recover the intended size. The [unsigned arithmetic reference][c-unsigned] describes that behavior. For a simple eight-bit illustration with $M=255$, a request for `100 * 4` becomes `144` after wraparound, even though 400 bytes were intended. Our guard instead compares `4 > 255 / 100`, which is true because integer division gives 2.

Start `buffer.c` with its includes and readable status names:

```c title="examples/02-storage/buffer.c — includes and status names"
#include "buffer.h"

#include <stdint.h>
#include <stdlib.h>

const char *storage_status_name(StorageStatus status)
{
    switch (status) {
    case STORAGE_OK:       return "ok";
    case STORAGE_ARGUMENT: return "invalid argument";
    case STORAGE_STATE:    return "invalid owner state";
    case STORAGE_OVERFLOW: return "size overflow";
    case STORAGE_ALLOC:    return "allocation failed";
    case STORAGE_RANGE:    return "view out of range";
    }
    return "unknown status";
}

```

`<stdint.h>` provides `SIZE_MAX`; `<stdlib.h>` declares `malloc` and `free`. The `switch` selects one named enum value. Each `case` returns immediately, so no `break` is needed and no case falls through. The returned strings have static storage duration; the caller neither frees nor changes them. An unrecognized value gets a diagnostic string rather than falling off the function.

Now append the checked multiplication and the default callback:

```c title="examples/02-storage/buffer.c — checked sizes and the system allocator"
StorageStatus size_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return STORAGE_ARGUMENT;
    }
    if (a != 0 && b > SIZE_MAX / a) {
        return STORAGE_OVERFLOW;
    }
    *out = a * b;
    return STORAGE_OK;
}

void *system_allocate(size_t bytes, void *context)
{
    (void)context;
    return malloc(bytes);
}

```

`&&` evaluates its right operand only when its left operand is true. Thus `a != 0 && ...` prevents division by zero. The assignment to `*out` happens only after all rejection paths; on `STORAGE_OVERFLOW` or `STORAGE_ARGUMENT`, the caller's scalar output is untouched.

The callback's `(void)context;` explicitly discards an unused parameter. `system_allocate` has the same two-argument type as the test callback even though `malloc` itself takes only a size. This tiny adapter avoids an incompatible function-pointer cast.

On the target, six floats require `6 * sizeof(float)` bytes, ordinarily 24. We nevertheless calculate with `sizeof(float)` instead of hardcoding a byte width. `SIZE_MAX` represents a type limit, not available RAM or a useful model size. A request can pass these arithmetic checks and still be rejected by the allocator.

The default `-fsanitize=undefined` group is not a replacement for this guard: unsigned wraparound is not signed-overflow undefined behavior. Clang lists unsigned-overflow checking separately in its [sanitizer reference][ubsan]. Our rejection is part of the program's normal behavior, including optimized builds without instrumentation.

## Construct locally, then commit the owner

Append the constructor:

```c title="examples/02-storage/buffer.c — the owner constructor"
StorageStatus buffer_create(FloatBuffer *out, size_t count,
                            AllocateFn allocate, void *context)
{
    if (out == NULL || allocate == NULL) {
        return STORAGE_ARGUMENT;
    }
    if (out->data != NULL || out->count != 0) {
        return STORAGE_STATE;
    }

    size_t bytes = 0;
    StorageStatus status = size_mul(count, sizeof(float), &bytes);
    if (status != STORAGE_OK) {
        return status;
    }
    if (count == 0) {
        return STORAGE_OK;
    }

    float *data = allocate(bytes, context);
    if (data == NULL) {
        return STORAGE_ALLOC;
    }
    *out = (FloatBuffer){.data = data, .count = count};
    return STORAGE_OK;
}

```

There are three boundaries before allocation. First, the output and callback must be non-null. Second, the output must already be the initialized empty owner. Third, its byte count must fit. A zero count succeeds without calling the allocator; the required non-null callback is still part of this interface even on that path. We avoid depending on the implementation-specific result of `malloc(0)`.

The local variable `data` receives the allocation result. In C, a returned `void *` converts to `float *` in this assignment; no cast is needed. A successful `malloc` provides storage, not initialized float values. The [allocation documentation][c-malloc] explains this distinction. Every element must receive a value before a numerical routine reads it. We deliberately do not conceal that responsibility in an automatic zeroing step.

Only after success does `*out = (FloatBuffer){...};` publish the new pointer and count together as a struct assignment. The expression `(FloatBuffer){...}` is a **compound literal**, a typed structure value whose named fields are initialized here. It is not a call to a constructor and does not allocate the six floats. See [compound-literal syntax][c-compound].

Consider the exact states for a six-float request:

| Point in the call | Local allocation | Caller owner |
|---|---|---|
| Entry | None | `{NULL, 0}` |
| Size check passes | None | `{NULL, 0}` |
| Callback returns `NULL` | None | Still `{NULL, 0}`; return allocation failure |
| Callback succeeds, before assignment | Locally held pointer to six uninitialized floats | Still `{NULL, 0}` |
| Final assignment | Responsibility transferred to `out` | Live owner with count 6 |

No operation after a successful allocation can report failure in this constructor. If later construction steps were added, a temporary owner would need cleanup until the final commit. Do not start by clearing `*out`: that would destroy an existing owner's pointer before reporting misuse. This version rejects reuse of a live owner instead of silently replacing it.

“Unchanged on failure” means the constructor does not write its output on any reported error. It does not mean the callback's state is unchanged—the failure probe intentionally increments a counter. It also does not promise recovery from an invalid dangling output pointer or a callback that violates its contract.

## Destroy exactly the object that owns the allocation

Append destruction:

```c title="examples/02-storage/buffer.c — destruction"
void buffer_destroy(FloatBuffer *buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    *buffer = (FloatBuffer){.data = NULL, .count = 0};
}

```

A null descriptor pointer is accepted as a no-op. Otherwise, `free(buffer->data)` releases the owned allocation; freeing a null allocation pointer is harmless. The assignment then restores the descriptor's empty state. Calling `buffer_destroy` twice on **this same initialized descriptor** is safe because its second call receives a null data pointer. This is not permission to free an arbitrary pointer twice. The [deallocation reference][c-free] explains that released storage is no longer available to the program through the old pointer.

The original allocation pointer must be passed to `free`, not the row pointer three elements into it. Our views expose no destruction function precisely because they have nothing to release. A view can stop being used without altering its owner.

A pointer field is only a value. Ordinary [structure assignment][c-struct-copy] copies that value, not the pointed-to allocation. The following is an ownership error, not a cloning operation:

```c title="Illustration — not part of the runnable program"
FloatBuffer duplicate = matrix; /* Two descriptors now contain one pointer. */
```

If both descriptors were treated as owners, destroying one would leave the other dangling; destroying the other would attempt a double free. Clearing `matrix.data` cannot reach into `duplicate` or any view and clear their copied pointers. A deliberate ownership transfer would need to leave the source empty and ensure the destination was previously empty. There is no automatic move operation in C, and this example does not need a transfer API.

Do not even inspect a stale view's pointer to decide whether its allocation is still alive. Its non-null value cannot answer that question. After owner destruction, stop using the view; to reuse the descriptor, overwrite it with a newly constructed valid view without reading its old pointer. Nor should one expect a visible drop in process memory after every `free`: allocator reuse and operating-system residency are separate from the logical ownership lifetime.

## Derive a strided range check without overflowing it

For a view, let $o$ be the starting offset within its owner, $n$ the number of selected elements, and $s$ its positive stride. Let $N$ be the owner's element count. Logical element $i$ maps to

$$
\operatorname{index}(i)=o+is,
\qquad 0\leq i<n.
$$

When $n>0$, the final selected index is $o+(n-1)s$. A direct test that calculates this expression first can itself wrap. We want the bound check to be safer than the access it protects.

After establishing $o<N$ and $s>0$, the validity condition can be rearranged:

$$
o+(n-1)s\leq N-1
\quad\Longleftrightarrow\quad
n-1\leq\left\lfloor\frac{N-1-o}{s}\right\rfloor.
$$

Each subtraction on the right is now safe because $o<N$, and division is safe because $s>0$. We have not formed a pointer or multiplied a huge stride. Row 1 has $N=6,o=3,n=3,s=1$: the test is $2\leq(5-3)/1=2$. Column 1 has $o=1,n=2,s=3$: the test is $1\leq\lfloor4/3\rfloor=1$.

Contrast an invalid two-element view starting at offset 3 with stride 3. It would select indices 3 and 6, but valid indices stop at 5. The rearranged check becomes $1\leq\lfloor2/3\rfloor=0$, which is false. The routine rejects it before constructing the view.

Append the complete builder:

```c title="examples/02-storage/buffer.c — borrow a validated view"
StorageStatus buffer_view(const FloatBuffer *owner, size_t offset,
                          size_t count, size_t stride, FloatView *out)
{
    if (owner == NULL || out == NULL || stride == 0) {
        return STORAGE_ARGUMENT;
    }
    if ((owner->data == NULL) != (owner->count == 0)) {
        return STORAGE_STATE;
    }
    if (offset > owner->count) {
        return STORAGE_RANGE;
    }
    if (count == 0) {
        *out = (FloatView){.data = NULL, .count = 0, .stride = 1};
        return STORAGE_OK;
    }
    if (offset == owner->count) {
        return STORAGE_RANGE;
    }
    const size_t remaining = owner->count - 1 - offset;
    if (count - 1 > remaining / stride) {
        return STORAGE_RANGE;
    }

    *out = (FloatView){.data = owner->data + offset,
                      .count = count, .stride = stride};
    return STORAGE_OK;
}

```

The first checks reject missing descriptors and a zero stride. We do not support repeated-element broadcasting, negative strides, or reverse traversal in this foundation interface. Such layouts require different contracts, not a cast of a negative number to `size_t`.

The comparison `(owner->data == NULL) != (owner->count == 0)` asks whether exactly one field describes an empty owner. A null pointer with a positive count, or a non-null pointer with a zero count, is inconsistent here. This catches a metadata error; it cannot establish that an otherwise plausible non-null pointer is live or really owns the advertised number of floats. Those remain caller obligations.

For an empty view, we allow offsets from zero through `owner->count`, then return the canonical descriptor `{NULL, 0, 1}`. Empty views select no elements, so they need not remember a location. In particular, an empty owner never causes `NULL + 0`, and an empty request never evaluates `count - 1`. Both branches must occur before the nonempty arithmetic.

For a nonempty view, the validated start is inside the allocation. Only after the quotient check do we calculate `owner->data + offset`. All later selected indices stay within that same allocation. Standard array/pointer bounds are relevant even when an address is never dereferenced: repeatedly advancing a strided pointer after its final use can form a pointer beyond one-past. The [GCC implementation reference][c-pointer-bounds] explicitly cautions that integer address arithmetic does not remove these constraints.

For example, column 1 starts at offset 1 and accesses offsets 1 and 4. Advancing that pointer once more by 3 would produce offset 7, beyond even the one-past offset 6. Instead, the sum below computes only the indices it actually accesses.

## Sum the view, not the gaps between its values

Finish `buffer.c`:

```c title="examples/02-storage/buffer.c — read a view"
float view_sum(FloatView view)
{
    float sum = 0.0f;
    for (size_t i = 0; i < view.count; ++i) {
        sum += view.data[i * view.stride];
    }
    return sum;
}
```

Passing `FloatView` by value copies three fields. It copies no float elements and acquires no ownership. For a view returned by the builder whose owner is still alive, each multiplication `i * view.stride` is bounded by the previously checked final offset. The loop never evaluates the expression for `i == view.count`.

| View | Loop index `i` | Offset from view start | Offset from owner start | Read | Running sum |
|---|---:|---:|---:|---:|---:|
| Row 1 | 0 | 0 | 3 | 4 | 4 |
| Row 1 | 1 | 1 | 4 | 5 | 9 |
| Row 1 | 2 | 2 | 5 | 6 | 15 |
| Column 1 | 0 | 0 | 1 | 2 | 2 |
| Column 1 | 1 | 3 | 4 | 5 | 7 |

An empty view returns `0.0f` without reading a pointer. A single-element view may have a very large positive stride, because its only access uses `i == 0`; the builder need not reject a stride that will never be used to step anywhere.

This reduction uses a float accumulator, in increasing logical-index order. Its small integer-valued fixtures and sums are exactly representable on the target, so equality is an appropriate test here. It is not a general tolerance policy for learned tensors. Floating-point rounding, non-finite values, and numerical error tests enter in Chapter 3.

`view_sum` is intentionally not another general validator. It assumes truthful metadata, initialized floats, and a live owner throughout the call. It neither allocates nor retains the view. Revalidating a count could not detect that a formerly valid allocation has been freed. Correct lifetime is a separate invariant from correct shape.

## Give the tests a private allocation counter

Begin `main.c` with the callback and two small test helpers:

```c title="examples/02-storage/main.c — callback and test helpers"
#include "buffer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t attempts;
    size_t fail_at;
} AllocationProbe;

static void *probe_allocate(size_t bytes, void *context)
{
    AllocationProbe *probe = context;
    ++probe->attempts;
    if (probe->fail_at != 0 && probe->attempts == probe->fail_at) {
        return NULL;
    }
    return malloc(bytes);
}

static int expect(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

static int same_view(FloatView a, FloatView b)
{
    return a.data == b.data && a.count == b.count && a.stride == b.stride;
}

```

At file scope, `static` on a function gives it internal linkage: this helper is private to `main.c`. It does not mean the helper's local variables persist between calls. Each test instead creates an `AllocationProbe` object and passes its address as `context`. The assignment `AllocationProbe *probe = context;` converts back to the type the caller supplied. The callback requires that object to remain alive and writable during the call.

`fail_at == 0` disables injection. Otherwise, a one-based attempt count selects exactly one call to return `NULL`. Rejected sizes and zero-length requests never enter this callback, so they never increment `attempts`. The tiny test driver makes only a handful of calls; this is not a concurrent allocation monitor or an unbounded counter service.

`expect` prints a failure message and returns the condition. Unlike placing work inside `assert`, these checks and their function calls remain present in optimized builds even if assertions are disabled. `same_view` compares the three field values, not the raw struct bytes, which may include padding. It is only used while both pointer fields still refer to live storage or are null.

The tests below use `goto cleanup;`. A `goto` transfers control to a named label **inside the same function**; it does not throw an exception or initialize variables that it skips. All resource owners are therefore initialized before any jump can reach cleanup. The cleanup block uses only those initialized owners and its initialized success flag. This focused forward jump keeps every error path from duplicating a different version of `free`. See [the `goto` reference][c-goto].

## Exercise ownership and views together

The next three blocks, in order, form the complete `test_views` function in `main.c`. The first creates and initializes the matrix:

```c title="examples/02-storage/main.c — test_views: construct and initialize"
static int test_views(void)
{
    int ok = 0;
    FloatBuffer matrix = {.data = NULL, .count = 0};
    FloatView row = {.data = NULL, .count = 0, .stride = 1};
    FloatView column = {.data = NULL, .count = 0, .stride = 1};
    FloatView empty = {.data = NULL, .count = 0, .stride = 1};
    AllocationProbe probe = {.attempts = 0, .fail_at = 0};
    size_t count = 0;

    if (!expect(size_mul(2, 3, &count) == STORAGE_OK, "matrix element count")) {
        goto cleanup;
    }
    StorageStatus status = buffer_create(&matrix, count, probe_allocate, &probe);
    if (!expect(status == STORAGE_OK, storage_status_name(status))) {
        goto cleanup;
    }
    for (size_t i = 0; i < matrix.count; ++i) {
        matrix.data[i] = (float)(i + 1);
    }

```

The test checks the element product `2 * 3` even though these particular constants are small. The constructor independently checks the resulting byte product. `&count`, `&matrix`, and `&probe` pass addresses of caller-owned objects; the functions may modify the designated outputs according to their contracts.

The six-element fill uses `(float)(i + 1)`. This explicit conversion documents that a small integer value becomes a float. Here `i` ranges from 0 to 5, so neither `i + 1` nor the conversion introduces a size or representability problem. Do not treat this fixture loop as a justification for converting arbitrarily large indices to float.

Append the two views and a controlled aliasing check:

```c title="examples/02-storage/main.c — test_views: sums and shared storage"
    if (!expect(buffer_view(&matrix, 3, 3, 1, &row) == STORAGE_OK,
                "borrow row 1") ||
        !expect(buffer_view(&matrix, 1, 2, 3, &column) == STORAGE_OK,
                "borrow column 1")) {
        goto cleanup;
    }
    const float row_total = view_sum(row);
    const float column_total = view_sum(column);
    printf("row 1 sum = %.0f\n", (double)row_total);
    printf("column 1 sum = %.0f\n", (double)column_total);
    if (!expect(row_total == 15.0f && column_total == 7.0f, "strided sums")) {
        goto cleanup;
    }

    FloatView copy = row;
    matrix.data[3] = 40.0f;
    if (!expect(view_sum(copy) == 51.0f && view_sum(row) == 51.0f,
                "view copies alias the original floats")) {
        goto cleanup;
    }
    matrix.data[3] = 4.0f;

```

The two view builders receive different start/count/stride triples but the same owner. No additional allocation occurs. A `FloatView` copy is allowed: it creates another borrowed description rather than another obligation to free memory.

Changing the owner's value at flat offset 3 from 4 to 40 changes the row sum from $4+5+6=15$ to $40+5+6=51$. Both `copy` and `row` observe 51. This is a direct check that views share bytes rather than snapshot values. The test restores 4 while the owner is still alive, so subsequent cases start with the original matrix. There is no concurrent mutation during `view_sum`.

Finish the function with rejected requests, owner protection, and cleanup:

```c title="examples/02-storage/main.c — test_views: rejected ranges and destruction"
    FloatView unchanged = row;
    if (!expect(buffer_view(&matrix, 3, 2, 3, &unchanged) == STORAGE_RANGE &&
                same_view(unchanged, row), "invalid last element preserves view") ||
        !expect(buffer_view(&matrix, 1, 2, SIZE_MAX, &unchanged) == STORAGE_RANGE &&
                same_view(unchanged, row), "huge stride rejected safely") ||
        !expect(buffer_view(&matrix, 0, 1, 0, &unchanged) == STORAGE_ARGUMENT &&
                same_view(unchanged, row), "zero stride rejected") ||
        !expect(buffer_view(&matrix, SIZE_MAX, 0, 1, &unchanged) == STORAGE_RANGE &&
                same_view(unchanged, row), "invalid empty offset rejected") ||
        !expect(buffer_view(&matrix, matrix.count, 0, 1, &empty) == STORAGE_OK &&
                empty.data == NULL && empty.count == 0 && empty.stride == 1 &&
                view_sum(empty) == 0.0f, "canonical empty view")) {
        goto cleanup;
    }

    float *original = matrix.data;
    const size_t attempts = probe.attempts;
    status = buffer_create(&matrix, 2, probe_allocate, &probe);
    if (!expect(status == STORAGE_STATE && matrix.data == original &&
                matrix.count == 6 && matrix.data[3] == 4.0f &&
                probe.attempts == attempts, "live owner cannot be overwritten")) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    /* No view, saved pointer, or owner copy is inspected after destruction. */
    buffer_destroy(&matrix);
    buffer_destroy(&matrix);
    if (!expect(matrix.data == NULL && matrix.count == 0, "destroy resets owner")) {
        ok = 0;
    }
    return ok;
}

```

`unchanged` starts as a valid copy of `row`, making failure preservation observable. Each failed builder must leave all three fields unchanged. The huge-stride case would be dangerous if the routine multiplied or added first; the division-based check rejects it without either operation. A zero stride is rejected even for layouts that might happen to touch valid memory, because it is outside this interface.

The empty view at offset `matrix.count` succeeds, but it is canonicalized to a null pointer; we do not read a one-past element. The constructor call on the already-live matrix must return `STORAGE_STATE` without calling the allocator or changing the existing data. `original` is a saved pointer for comparison **before** destruction, not another owner.

`ok` begins as false. Only reaching the end of all checks makes it true. Every error jumps to the same cleanup block, which releases the matrix even if a later test failed. Repeating destruction verifies that reset works on that descriptor. The final comparison reads the now-empty `matrix`, not `original`, `row`, `copy`, or `column`. Those borrowed pointers are no longer evaluated.

## Reject impossible requests before the allocator sees them

The following two blocks form `test_sizes_and_failures`. Begin with arithmetic boundaries and the oversized buffer request:

```c title="examples/02-storage/main.c — test_sizes_and_failures: overflow"
static int test_sizes_and_failures(void)
{
    int ok = 0;
    size_t product = 77;
    FloatBuffer candidate = {.data = NULL, .count = 0};
    FloatView empty = {.data = NULL, .count = 0, .stride = 1};
    AllocationProbe probe = {.attempts = 0, .fail_at = 1};

    if (!expect(size_mul(SIZE_MAX, 2, &product) == STORAGE_OVERFLOW &&
                product == 77, "overflow leaves scalar output unchanged") ||
        !expect(size_mul(0, SIZE_MAX, &product) == STORAGE_OK && product == 0,
                "zero product") ||
        !expect(size_mul(1, SIZE_MAX, &product) == STORAGE_OK && product == SIZE_MAX,
                "maximum representable product") ||
        !expect(size_mul(2, 3, NULL) == STORAGE_ARGUMENT, "null size output")) {
        goto cleanup;
    }
    if (!expect(sizeof(float) > 1, "byte-overflow fixture needs sizeof(float) > 1")) {
        goto cleanup;
    }
    StorageStatus status = buffer_create(&candidate, SIZE_MAX, probe_allocate, &probe);
    if (!expect(status == STORAGE_OVERFLOW && probe.attempts == 0 &&
                candidate.data == NULL && candidate.count == 0,
                "byte overflow rejected before allocator")) {
        goto cleanup;
    }
    puts("overflow rejected before allocation");

```

The scalar output starts at 77. After the overflowing multiplication it must still be 77; after `0 * SIZE_MAX` it must be zero; and `1 * SIZE_MAX` is representable. These are tests of the arithmetic helper, not attempts to allocate any such amount.

The byte-overflow fixture requests `SIZE_MAX` **floats**, not bytes. On the target, `sizeof(float)` is greater than one, so the byte product cannot fit. We check that fixture assumption explicitly instead of claiming every conceivable C implementation has four-byte floats. The allocator attempt count must still be zero. The probe is also configured to reject its first actual call, so a broken size guard would fail a check rather than intentionally consume the machine's memory.

Append the empty and injected-failure cases, then close the function:

```c title="examples/02-storage/main.c — test_sizes_and_failures: empty and failed construction"
    status = buffer_create(&candidate, 0, probe_allocate, &probe);
    if (!expect(status == STORAGE_OK && probe.attempts == 0 &&
                candidate.data == NULL && candidate.count == 0, "empty owner") ||
        !expect(buffer_view(&candidate, 0, 0, 1, &empty) == STORAGE_OK &&
                view_sum(empty) == 0.0f, "empty owner view without pointer arithmetic")) {
        goto cleanup;
    }
    status = buffer_create(&candidate, 6, probe_allocate, &probe);
    if (!expect(status == STORAGE_ALLOC && probe.attempts == 1 &&
                candidate.data == NULL && candidate.count == 0,
                "injected constructor failure preserves empty owner")) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    buffer_destroy(&candidate);
    buffer_destroy(NULL);
    return ok;
}

```

An empty construction leaves the candidate empty and the attempt counter at zero. An empty view of that owner returns sum zero without pointer arithmetic. The next six-float request reaches the callback, increments `attempts` to one, and receives `NULL`; the output owner remains empty. Cleanup can therefore use the same destroy function on either success or failure.

Do not try to prove this behavior with only `candidate.data == NULL`: a broken constructor could leave `count == 6` beside the null pointer. The test inspects both fields. Our owner invariant is about the pair.

## Clean up a partially acquired workspace

One allocation failure is easy: there is no allocation to release. The important next case is **second allocation failure after the first succeeded**. A numerical application repeatedly encounters this when it acquires parameters, intermediate arrays, and scratch space.

`matrix_sums` below accepts a writable two-float output array. On success it returns `[15, 7]`; on a reported failure it leaves both caller values unchanged. Its two owners are a six-float matrix and a two-float scratch buffer. Two stack floats would be sufficient for these particular sums; the heap scratch is deliberately present to expose the second-resource cleanup path, not because it improves this tiny computation.

Both owners and both view descriptors are initialized before any jump. Begin the function by validating the output and acquiring resources:

```c title="examples/02-storage/main.c — matrix_sums: acquire resources"
static StorageStatus matrix_sums(float *out, AllocateFn allocate, void *context)
{
    FloatBuffer matrix = {.data = NULL, .count = 0};
    FloatBuffer scratch = {.data = NULL, .count = 0};
    FloatView row = {.data = NULL, .count = 0, .stride = 1};
    FloatView column = {.data = NULL, .count = 0, .stride = 1};
    StorageStatus status = STORAGE_ARGUMENT;

    if (out == NULL || allocate == NULL) {
        goto cleanup;
    }
    status = buffer_create(&matrix, 6, allocate, context);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
    status = buffer_create(&scratch, 2, allocate, context);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
```

If the first allocation fails, both owners remain empty. If the second fails, `matrix` is live and `scratch` is empty. In either case the status already records the reason, and `goto cleanup` bypasses computation and publication of results.

The pointer `out` must have room for two float elements; as in Chapter 1, the C pointer itself carries no capacity. The callback follows the same fresh-allocation rule as before and may not mutate the caller result indirectly. Null is a reported argument error; a forged short output pointer would violate the interface rather than become a safe error.

Append initialization, view construction, result publication, and the one cleanup path:

```c title="examples/02-storage/main.c — matrix_sums: compute, commit, and release"
    for (size_t i = 0; i < matrix.count; ++i) {
        matrix.data[i] = (float)(i + 1);
    }
    status = buffer_view(&matrix, 3, 3, 1, &row);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
    status = buffer_view(&matrix, 1, 2, 3, &column);
    if (status != STORAGE_OK) {
        goto cleanup;
    }
    scratch.data[0] = view_sum(row);
    scratch.data[1] = view_sum(column);
    out[0] = scratch.data[0];
    out[1] = scratch.data[1];

cleanup:
    buffer_destroy(&scratch);
    buffer_destroy(&matrix);
    return status;
}

```

Every possible reported failure occurs before writing `out[0]` or `out[1]`. The scratch values are fully initialized before being read. On the successful path, the last view constructor has left `status == STORAGE_OK`; the subsequent sums, assignments, and valid destruction operations do not introduce another status failure.

The cleanup order reverses acquisition: scratch first, matrix second. Neither view is read after matrix destruction. The status is returned after cleanup instead of being replaced with success merely because cleanup finished.

This is a **failure-preserving commit convention**, not a claim of atomicity for multiple threads or signals. The two output assignments are separate C operations. A caller that concurrently observes or changes these buffers needs an entirely different synchronization contract. This foundation program is single-threaded.

| Failure selection | Matrix at cleanup entry | Scratch at cleanup entry | Caller output after return |
|---|---|---|---|
| First allocation | Empty | Empty | Original two values |
| Second allocation | Live | Empty | Original two values |
| No injected failure | Live | Live | `[15, 7]` |

In all three cases, both local owners are empty by the time the function returns. A single cleanup path makes that state transition visible. It does not remove the need to initialize every owner that cleanup may touch.

Append the driver that selects both failure sites and then retries without injection:

```c title="examples/02-storage/main.c — test both allocation failures"
static int test_partial_cleanup(void)
{
    for (size_t fail_at = 1; fail_at <= 2; ++fail_at) {
        AllocationProbe probe = {.attempts = 0, .fail_at = fail_at};
        float out[2] = {-7.0f, -9.0f};
        StorageStatus status = matrix_sums(out, probe_allocate, &probe);
        if (!expect(status == STORAGE_ALLOC && probe.attempts == fail_at &&
                    out[0] == -7.0f && out[1] == -9.0f,
                    "partial construction preserves caller result")) {
            return 0;
        }
        printf("allocation %zu failure: caller result unchanged\n", fail_at);
    }
    float out[2] = {-7.0f, -9.0f};
    StorageStatus status = matrix_sums(out, system_allocate, NULL);
    if (!expect(status == STORAGE_OK && out[0] == 15.0f && out[1] == 7.0f,
                "success after injected failures")) {
        return 0;
    }
    printf("success after failure: [%.0f, %.0f]\n", (double)out[0], (double)out[1]);
    return 1;
}

```

The caller output begins at `[-7, -9]`, values different from either successful result. A zero-filled fixture would be weaker if a broken routine also wrote zeros on failure. Each loop iteration uses a fresh probe, so failure number 2 really means the second allocation of that attempt, not the second allocation of the whole process.

With `fail_at == 2`, the first callback delegates to `malloc`, and the second returns `NULL`. The output checks prove preservation and the call counter proves which boundary was reached. The code's cleanup path releases the first allocation. A leak-enabled sanitizer run supplies an additional check that this path did not leave an allocation behind; absence of a leak report is evidence about that run, not a proof of all future paths.

The final call uses `system_allocate`, not an injected failure. Success afterward confirms that failure state was local rather than left in a global “allocator broken” switch. A genuine machine allocation failure can still cause this positive test to fail; injection does not make allocation infallible.

## Distinguish aliasing from a restricted access promise

Two pointers **alias** when their accesses refer to the same element. Our row and column views intentionally alias at matrix element 5, flat offset 4. Two read-only sums can share that element without a problem. Likewise, `const` does not assert that two pointers are disjoint.

`restrict` supplies a different kind of information. For the function below, the written output range must be disjoint from both read input ranges during the call. The compiler may optimize using that promise; no runtime overlap test is inserted. A local pointer derived from a restricted pointer can still be used to access its elements—`restrict` is not a ban on copying a pointer value within its permitted access path.

Append a small function whose intended promise is explicit:

```c title="examples/02-storage/main.c — a deliberately disjoint-output kernel"
/* out[0..n) must not overlap either input. Read-only inputs may overlap. */
static void add_disjoint(float *restrict out, const float *restrict a,
                         const float *restrict b, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

```

The function reads `a` and `b` and writes `out`. Its two inputs may be the **same** array because their elements are only read and do not overlap the written output. This read/read distinction is important: `restrict` does not simply mean all parameter addresses must differ. LLVM's [clarification of its C-derived aliasing rule][llvm-restrict] discusses why the restriction depends on modified locations, and the GNU manual gives an [input/output optimization example][c-restrict].

Conversely, calling this function with `out == a` for a positive length violates its contract even if a scalar version seems to double elements correctly. Partially overlapping ranges such as an output beginning one element into an input also violate it. Different starting addresses alone do not establish disjointness. Do not treat a quiet AddressSanitizer run as permission: an aliasing-promise violation can concern entirely allocated memory.

Use `restrict` only after choosing and documenting the actual access contract. It is neither ownership nor bounds checking nor a synchronization primitive, and it promises no automatic speedup. The storage/view functions do not need it. Later matrix kernels will revisit overlap rules with explicit input, output, and accumulation shapes.

Finish `main.c` with the test sequence and a valid shared-input call:

```c title="examples/02-storage/main.c — main"
int main(void)
{
    if (!test_views() || !test_sizes_and_failures() || !test_partial_cleanup()) {
        return EXIT_FAILURE;
    }
    const float input[3] = {1.0f, 2.0f, 3.0f};
    float doubled[3] = {0.0f, 0.0f, 0.0f};
    add_disjoint(doubled, input, input, 3);
    if (!expect(doubled[0] == 2.0f && doubled[1] == 4.0f && doubled[2] == 6.0f,
                "restricted output with shared read-only inputs")) {
        return EXIT_FAILURE;
    }
    puts("storage checks passed");
    return EXIT_SUCCESS;
}
```

`input` and `doubled` are separate arrays. Passing `input` for both read arguments is valid, and the exact result is `[2, 4, 6]`. The tests return failure immediately when a prerequisite test fails; each resource-owning test has already run its cleanup before returning. No test framework or assertion macro conceals that control flow.

## Build the complete example in three configurations

Create the full Makefile below. Recipe lines must begin with a tab. It uses the same established build structure as Chapter 1, with this chapter's two source files and a distinct executable name:

```makefile title="examples/02-storage/Makefile"
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
```

The debug build favors inspection; the sanitizer build adds address and undefined-behavior instrumentation at compilation **and linking**; the optimized build does not depend on sanitizer checks for correctness. Each configuration has its own object files, so a previous debug object cannot silently become part of the release executable. The Makefile itself is a dependency, so changing its flags triggers the relevant rebuild.

From `examples/02-storage/` on the native Mac toolchain, run:

```sh title="Terminal — build and run all configurations"
make test
```

A compatible alternative compiler can be selected explicitly, for example `make CC=clang test`. No new package, exact compiler version, or macOS version is required beyond the C17 and sanitizer capabilities established in Chapter 1.

Ignoring compiler command lines, each executable should print the same seven lines and exit with status zero:

```text title="Expected program output — once per configuration"
row 1 sum = 15
column 1 sum = 7
overflow rejected before allocation
allocation 1 failure: caller result unchanged
allocation 2 failure: caller result unchanged
success after failure: [15, 7]
storage checks passed
```

This output includes checks for arithmetic overflow, canonical emptiness, unchanged construction results, invalid views, copying views without copying floats, live-owner replacement rejection, repeated destruction of the same reset owner, both allocation failures, and the valid `restrict` call. Printing a correct row sum alone would not establish any of those other properties.

The accompanying source has also been run with Clang 17 on x86_64 Linux in all three configurations. That is a cross-platform source check, not execution on the target M4 Mac, an observed macOS allocator trace, or a performance result. Run the same commands on your machine.

Where the selected sanitizer runtime supports leak detection, enable it for an additional run:

```sh title="Terminal — leak check with a supporting sanitizer runtime"
ASAN_OPTIONS=detect_leaks=1 ./build/sanitize/storage-demo
```

Clang's [AddressSanitizer documentation][asan] describes platform support and report symbolization. Runtime availability can differ between compiler distributions. An unsupported leak option is not a numerical test failure, and turning on a sanitizer is not a substitute for identifying the owner on every control-flow path.

## Isolate a dangling-view error instead of normalizing it

The normal tests must stay defined and passing. To observe a use-after-free diagnostic, create a separate fault executable under `build/faults/`; it is never part of `make test`.

```sh title="Terminal — create only the temporary diagnostic directory"
mkdir -p build/faults
```

Create the complete temporary source below. It correctly constructs, initializes, and borrows a row, but then deliberately destroys the owner too early:

```c title="examples/02-storage/build/faults/use_after_free.c — intentional fault"
#include "buffer.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    FloatBuffer owner = {.data = NULL, .count = 0};
    FloatView row = {.data = NULL, .count = 0, .stride = 1};
    if (buffer_create(&owner, 6, system_allocate, NULL) != STORAGE_OK) {
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < owner.count; ++i) {
        owner.data[i] = (float)(i + 1);
    }
    if (buffer_view(&owner, 3, 3, 1, &row) != STORAGE_OK) {
        buffer_destroy(&owner);
        return EXIT_FAILURE;
    }
    buffer_destroy(&owner);
    printf("invalid row sum = %g\n", (double)view_sum(row));
    return EXIT_SUCCESS;
}
```

Build this file with `buffer.c`, **not** `main.c`, to avoid linking two definitions of `main`:

```sh title="Terminal — intentionally failing sanitizer executable"
xcrun clang -std=c17 -Wall -Wextra -Wpedantic -O1 -g \
  -fno-omit-frame-pointer -fsanitize=address,undefined \
  -fno-sanitize-recover=undefined -I. \
  buffer.c build/faults/use_after_free.c -o build/faults/use-after-free
./build/faults/use-after-free
```

Expect a nonzero exit and a `heap-use-after-free` report, with the invalid read in `view_sum` and allocation/free call sites when symbols are available. Do not expect a meaningful “invalid row sum,” a fixed address, or a particular source line number after editing. The important trace is: allocation in the constructor, release in destruction, read through the stale view. The owner descriptor being empty does not rescue the separately copied pointer in `row`.

Repair this exercise by computing and printing the sum **before** destroying the owner; then both operations refer to live storage at the time they use it. Rebuilding the isolated executable should print `invalid row sum = 15` without a lifetime report—the old label is merely diagnostic text, so rename it to `row sum` as part of the repair. Do not “repair” the code by testing whether `row.data` is non-null or by suppressing the sanitizer.

Remove the temporary diagnostic and rebuild the normal example:

```sh title="Terminal — restore the clean prerequisite state"
make clean
make test
```

The fault source lives under `build/`, so `clean` removes it as well as its executable. The four persistent files remain unchanged. AddressSanitizer catches this particular lifetime error; it cannot establish every ownership rule or detect every wrong-but-in-bounds layout.

## Make two small predictions, then change the program

**Change a stride without leaving the allocation.** In the successful column construction inside `test_views`, replace the stride 3 with 1. The view now selects offsets 1 and 2, values 2 and 3, so the printed column sum becomes 5. The existing `strided sums` check must fail, but the memory sanitizer should find no out-of-bounds access. This is a layout error, not a lifetime error. Restore stride 3 afterward.

**Test a one-element view with a large unused stride.** In `test_views`, insert the following immediately before the declaration of `original`, while `matrix` is still live:

```c title="Variation — insert inside test_views, then restore the baseline"
    FloatView last = {.data = NULL, .count = 0, .stride = 1};
    if (!expect(buffer_view(&matrix, 5, 1, SIZE_MAX, &last) == STORAGE_OK &&
                view_sum(last) == 6.0f, "one element does not step by its stride")) {
        goto cleanup;
    }
```

The quotient check accepts this: `count - 1` is zero. The only index product in the sum is `0 * SIZE_MAX`, also zero. Change the requested count to 2, and the builder must reject it before any access. Change the expected status for that negative experiment instead of calling `view_sum` on an unconstructed result. Remove the extra case after the experiment to retain the chapter's named baseline.

These changes should be small. Their purpose is to distinguish the logical values selected by a layout, the arithmetic safety of an index calculation, and the lifetime of the allocation—three related but different questions.

## Diagnose the failing invariant

| Symptom | Useful first inspection |
|---|---|
| `STORAGE_STATE` on the first construction | Was the descriptor initialized to `{NULL, 0}` before the call? Reading an uninitialized descriptor is already a bug. |
| `STORAGE_STATE` when resizing | This constructor does not resize or replace a live owner. Destroy it at a valid lifetime boundary, then construct anew; old views cannot survive that destruction. |
| Allocation count changes during an overflow test | Check that both size multiplications are validated before invoking the callback. |
| Second-allocation failure leaks memory | Follow the jump to `cleanup`; the first owner must still be reachable and destroyed there. |
| A view builder unexpectedly returns `STORAGE_RANGE` | Write down the selected offsets and the owner's count. Keep stride in elements, not bytes. |
| Row sum is correct but column sum is wrong | Inspect the column stride and start offset; a valid pointer is not proof of the intended shape. |
| Double-free or invalid-free report | Look for an owner copied by assignment, an interior view pointer passed to `free`, or fields overwritten before cleanup. |
| Use-after-free with a non-null view | The view carries no lifetime protection. Move its last use before owner destruction, not merely behind a null check. |
| Debug works but optimized output changes | Check initialization and overlap contracts before blaming optimization. In particular, `restrict` promises are not checked dynamically. |

The interface deliberately reports invalid sizes and shapes while leaving actual pointer validity as a caller obligation. Do not manufacture invalid pointers merely to see which status comes back. Negative tests use valid initialized descriptors and intentionally bad **metadata requests**, not undefined memory accesses in the normal executable.

## Carry the storage rules into the numerical foundation

You now have an independently runnable storage module: empty/live owner states, checked byte counts, allocation-free views, explicit failure statuses, deterministic allocation failure, and one cleanup path for a two-resource computation. The complete source is sufficient without the decoder tree or a hidden utility library.

The rules that will matter for the model are concrete. Counts and strides have units. A returned view does not copy elements or extend their lifetime. Saved intermediates must remain alive until their last reader. A failed constructor must not lose the caller's prior state. Clearing an owner does not invalidate copies on its behalf; those aliases must already be out of use.

Chapter 3 turns to the values stored in these buffers: floating-point rounding, cancellation, stable exponentials, and numerical derivatives. Chapter 4 will then combine this storage vocabulary with actual tensor layouts and matrix products. Keep this foundation example separate and passing while those later modules are introduced.

## References

**C declarations and values:** GNU C Language Manual, [array subscripting and pointer conversion][c-arrays], [designated initializers][c-designated], [structure assignment][c-struct-copy], [compound literals under “Structure Constructors”][c-compound], [function-pointer declarations][c-function-pointer], and [forward cleanup jumps using `goto`][c-goto]. The example uses standard C17 constructs; GNU-only extensions appearing elsewhere in that manual are not required.

**Size, allocation, and lifetime:** GNU C Language Manual, [unsigned arithmetic][c-unsigned]; GNU C Library Reference Manual, [basic allocation][c-malloc] and [freeing allocated storage][c-free]. These references explain the used C interfaces; the program does not require glibc on macOS or rely on glibc-specific allocator internals. GCC, [array/pointer implementation notes][c-pointer-bounds], for the limits of pointer arithmetic.

**Access promises and diagnostics:** GNU C Language Manual, [`restrict` input/output example][c-restrict]; LLVM's [review D74935][llvm-restrict], specifically its clarification that read/read aliasing differs from access to modified locations; Clang, [AddressSanitizer usage, symbolization, and leak support][asan] and [UndefinedBehaviorSanitizer's available checks][ubsan].

[c-arrays]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Pointers-and-Arrays.html
[c-designated]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Designated-Inits.html
[c-struct-copy]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Structure-Assignment.html
[c-compound]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Structure-Constructors.html
[c-function-pointer]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Declaring-Function-Pointers.html
[c-goto]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/goto-Statement.html
[c-unsigned]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Unsigned-Overflow.html
[c-malloc]: https://sourceware.org/glibc/manual/latest/html_node/Basic-Allocation.html
[c-free]: https://sourceware.org/glibc/manual/latest/html_node/Freeing-after-Malloc.html
[c-pointer-bounds]: https://gcc.gnu.org/onlinedocs/gcc/Arrays-and-pointers-implementation.html
[c-restrict]: https://www.gnu.org/software/c-intro-and-ref/manual/html_node/restrict-Pointer-Example.html
[llvm-restrict]: https://reviews.llvm.org/D74935
[asan]: https://clang.llvm.org/docs/AddressSanitizer.html
[ubsan]: https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html#available-checks
[source-header]: https://25d-228.github.io/gpt-c/media/chapter-02/reference/buffer.h
[source-buffer]: https://25d-228.github.io/gpt-c/media/chapter-02/reference/buffer.c
[source-main]: https://25d-228.github.io/gpt-c/media/chapter-02/reference/main.c
[source-make]: https://25d-228.github.io/gpt-c/media/chapter-02/reference/Makefile

