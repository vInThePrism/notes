# CuTe DSL: Printing, Debugging, and Layouts

## Overview

This note covers how to print and debug values in CuTe DSL, with a focus on understanding the critical distinction between **static (compile-time)** and **dynamic (runtime)** values. It also explains **CuTe layouts** — the shape+stride system that controls how data is accessed in memory.

---

## 1. Two Printing Mechanisms

CuTe JIT compilation has two phases, and printing behaves differently in each:

| Method | Executes At | Can Show Dynamic Values? |
|---|---|---|
| `print()` (Python built-in) | Compile time (tracing) | No — prints `?` for dynamic values |
| `cute.printf()` | Runtime (on device) | Yes — prints actual values |

### Example

```python
@cute.jit
def print_example(a: cutlass.Int32, b: cutlass.Constexpr[int]):
    print(">>>", b)            # Compile-time: outputs 2
    print(">>>", a)            # Compile-time: outputs ? (a is dynamic)
    cute.printf(">?? {}", a)   # Runtime: outputs 8
    cute.printf(">?? {}", b)   # Runtime: outputs 2
```

### F-string Trap

F-strings are evaluated at compile time, so dynamic values always show as `?`:

```python
print(f"a: {a}, b: {b}")  # => "a: ?, b: 2" — even at runtime
```

**Rule:** Always use `cute.printf()` when you need to see runtime values.

### Compilation vs Execution

- `cute.compile(fn, ...)` — only traces, triggers `print()` output (static info)
- Calling the compiled result — only triggers `cute.printf()` output (dynamic info)
- Calling `fn(...)` directly — does both in sequence

---

## 2. Layouts: Shape + Stride

A CuTe layout describes how multi-dimensional data maps to linear memory. Format: `(shape):(stride)`.

### What Shape and Stride Mean

- **Shape** — dimensions of the logical tensor
- **Stride** — for each dimension, how many elements to skip in memory when moving one step along that dimension

Shape and stride are paired per-dimension. A 2D tensor has 2 shape values and 2 stride values; a 3D tensor has 3 of each.

### Concrete Example

`(2,3):(3,1)` means:

- Shape: 2 rows, 3 columns
- Stride: moving one row jumps 3 elements; moving one column jumps 1 element

Memory offset formula: `offset = i * 3 + j * 1`

```
Memory: [0, 1, 2, 3, 4, 5]

         j=0   j=1   j=2
i=0:      0     1     2
i=1:      3     4     5
```

### Row-Major vs Column-Major

Same data, different layouts:

```
Row-major    (2,3):(3,1)        Column-major  (2,3):(1,2)
  0  1  2                         0  2  4
  3  4  5                         1  3  5
```

The memory contents are identical — only the stride changes, and the logical matrix transposes.

### Static vs Dynamic in Layouts

When building a layout from mixed static/dynamic values:

```python
layout = cute.make_layout((a, b))  # a=Int32 (dynamic), b=Constexpr (static)
```

| Component | Compile-time | Runtime |
|---|---|---|
| Shape dim 0 | `?` (a unknown) | `8` |
| Shape dim 1 | `2` (b is const) | `2` |
| Stride dim 0 | `1` (fixed) | `1` |
| Stride dim 1 | `?` (depends on a) | `8` |

Compile-time print: `(?,2):(1,?)` — Runtime print: `(8,2):(1,8)`

---

## 3. Why Layouts Matter for Performance

### Only One Dimension Can Have stride=1

Data in memory is a flat 1D array. Only one direction of traversal can be truly contiguous (stride=1). Two dimensions with stride=1 would mean different coordinates map to the same memory location.

### The Stride=1 Rule

**Make your innermost/most-frequent loop dimension the one with stride=1.** This ensures sequential memory access.

- Row-major `(M,N):(N,1)` — scanning along columns (dim 1) is contiguous
- Column-major `(M,N):(1,M)` — scanning along rows (dim 0) is contiguous

### GPU-Specific: Coalesced Access

On GPUs, adjacent threads in a warp should access adjacent memory addresses. This means the dimension indexed by thread ID should be the stride=1 dimension. Misalignment here can cause **10x+ performance degradation**.

### Matrix Multiply Tiling Example

When tiling a matrix multiply, layout choices cascade through every stage:

```
Global memory:  (M, K):(K, 1)     — original matrix layout
Tile:           (64, 32):(32, 1)  — a block cut from the matrix
Shared memory:  (64, 32):(1, 64)  — may change stride to avoid bank conflicts
```

Wrong stride at any stage means either incorrect data or severe performance loss (bank conflicts can slow shared memory access by ~32x).

### Static vs Dynamic: Performance Implications

Static layouts (all dimensions known at compile time) allow the compiler to:
- Unroll loops
- Pre-compute address offsets
- Eliminate redundant instructions

When you see `?` in a layout during debugging, ask: can this dimension be made into a compile-time constant? If yes, performance may improve.

---

## 4. Printing Tensors

`cute.print_tensor()` is a dedicated tensor visualization tool.

### Output Format

```
tensor(raw_ptr(0x...address: f32, generic, align<4>) o (4,3):(3,1), data=
       [[ 0.0,  1.0,  2.0, ],
        [ 3.0,  4.0,  5.0, ],
        ...])
```

Components: pointer address, data type, storage space (generic/gmem/rmem), layout, and torch-style data display.

### Three Modes

**Basic:** `cute.print_tensor(x)` — standard visualization.

**Verbose:** `cute.print_tensor(x, verbose=True)` — shows coordinate-to-value mapping for every element:

```
(0,0)= 0.0
(0,1)= 1.0
(1,0)= 3.0
...
```

**Sliced:** Use `cute.slice_()` to extract a sub-tensor, load into register memory (rmem), then print:

```python
sliced = cute.slice_(x, (None, 0))   # first column
sliced = cute.slice_(x, (1, None))   # second row
```

### GPU Tensor Printing

Use `cute.print_tensor()` inside a `@cute.kernel`. Launch with single thread `grid=(1,1,1), block=(1,1,1)` to avoid interleaved output from multiple threads. Storage space shows as `gmem` instead of `generic`.

### Supported Data Types

Integer types and `Float16`/`Float32`/`Float64`. More types planned for future releases.
