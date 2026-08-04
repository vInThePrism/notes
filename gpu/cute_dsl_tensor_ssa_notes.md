# TensorSSA in CuTe DSL

## What is TensorSSA?

TensorSSA is a Python class in CuTe DSL that represents a **tensor value residing in GPU registers**, expressed in Static Single Assignment (SSA) form.

### Breaking Down the Name

**SSA (Static Single Assignment)**: A compiler IR convention where every variable is assigned exactly once. Instead of mutating `x = 1; x = x + 2`, the compiler internally creates `x1 = 1; x2 = x1 + 2`. This makes optimization (dead code elimination, constant propagation, vectorization) straightforward because data flow is unambiguous.

**"Register tensor"**: GPU data lives at different memory hierarchy levels — Global Memory (GMEM, ~80GB, slow) → Shared Memory (SMEM, ~228KB/SM, fast) → Registers (RMEM, ~256 per thread, fastest). TensorSSA represents data that has been loaded into registers — the only place where the ALU can actually perform arithmetic.

### The Core Data Flow

```
cute.Tensor (address + layout in memory)
      │
   .load()  →  issues a memory read instruction (e.g. LDG on NVIDIA GPUs)
      │
  TensorSSA (data now lives in registers, wrapped as a Python object)
      │
   arithmetic, slicing, reduction, broadcast ...
      │
   .store()  →  issues a memory write instruction (e.g. STG)
      │
cute.Tensor (result written back to memory)
```

### Analogy

- `cute.Tensor` = a library call number ("Shelf A, Row 3, Slot 7")
- `.load()` = walking over and picking the book up onto your desk
- `TensorSSA` = the open book on your desk — you can read it, annotate it, compare with another book
- `.store()` = putting your notes back on the shelf

You cannot "add two call numbers together." You must physically retrieve the data first.

### SSA in Practice

Every operation produces a **new** TensorSSA; none are mutated:

```python
a_vec = a.load()        # TensorSSA #1
b_vec = b.load()        # TensorSSA #2
c_vec = a_vec + b_vec   # TensorSSA #3 (new; a_vec and b_vec unchanged)
d_vec = c_vec * 2.0     # TensorSSA #4 (another new value)
```

Each line corresponds to a new SSA value node in the MLIR IR that CuTe DSL generates. The compiler sees a clean computation graph with no aliasing ambiguity.

---

## Why TensorSSA Exists

It solves one problem: **making register-level GPU tensor operations feel like Python/NumPy.**

Under the hood, CuTe DSL generates MLIR IR that compiles down to optimized vectorized GPU instructions. But the user just writes `a_vec + b_vec`. TensorSSA achieves this by overloading Python operators (`+`, `-`, `*`, `/`, `[]`, etc.) and translating them into the corresponding IR operations.

---

## Core Usage: Load → Compute → Store

```python
@cute.jit
def load_and_store(res: cute.Tensor, a: cute.Tensor, b: cute.Tensor):
    a_vec = a.load()           # cute.Tensor → TensorSSA
    b_vec = b.load()
    res.store(a_vec + b_vec)   # TensorSSA → cute.Tensor
```

| Object | Type | Lives in | Can compute directly? |
|--------|------|----------|-----------------------|
| `a`, `b`, `res` | `cute.Tensor` | Memory (GMEM/SMEM) | No — just an address + layout descriptor |
| `a_vec`, `b_vec` | `TensorSSA` | Registers | Yes — supports all arithmetic |

Printing `a_vec` shows `vector<12xf32> o (3, 4)`: a 12-element f32 vector with logical shape (3, 4).

---

## Slicing

TensorSSA supports NumPy-like indexing:

```python
src_vec = src.load()             # shape (4, 2, 3)
dst_vec = src_vec[(None, 1, None)]  # equivalent to src[:, 1, :]
# Result shape: (4, 3)
```

- `None` = keep this dimension (take all elements)
- An integer = select that index along this dimension (dimension is removed)

With a scalar index like `src_vec[10]`, the result is a scalar (not a TensorSSA), so the code uses `isinstance(dst_vec, cute.TensorSSA)` to branch accordingly.

---

## Arithmetic Operations

All TensorSSA operations are **element-wise**.

### Binary: TensorSSA ⊕ TensorSSA

```python
a_vec = [1.0, 1.0, 1.0]
b_vec = [2.0, 2.0, 2.0]

a_vec + b_vec   # [3, 3, 3]
a_vec - b_vec   # [-1, -1, -1]
a_vec * b_vec   # [2, 2, 2]
a_vec / b_vec   # [0.5, 0.5, 0.5]
a_vec // b_vec  # [0, 0, 0]   (floor division)
a_vec % b_vec   # [1, 1, 1]   (modulo)
```

### Binary: TensorSSA ⊕ Scalar

The scalar is **automatically broadcast** to match the TensorSSA shape:

```python
a_vec + 2.0   # equivalent to a_vec + [2.0, 2.0, 2.0]
```

The scalar is passed as `cutlass.Constexpr` (a compile-time constant).

### Comparison (returns boolean TensorSSA)

```python
a_vec = [1, 2, 3];  b_vec = [2, 1, 4]
a_vec > b_vec    # [False, True, False]
# Also: >=, <, <=, ==
```

### Bitwise (integer types only)

```python
a_vec = [1, 2, 3]   # binary: 001, 010, 011
b_vec = [2, 2, 4]   # binary: 010, 010, 100

a_vec ^ b_vec   # XOR → [3, 0, 7]
a_vec | b_vec   # OR  → [3, 2, 7]
a_vec & b_vec   # AND → [0, 2, 0]
```

### Unary (`cute.math.*`)

```python
a_vec = [4.0, 4.0, 4.0]

cute.math.sqrt(a_vec)   # [2.0, 2.0, 2.0]
cute.math.sin(a_vec)    # [-0.7568, ...]
cute.math.exp2(a_vec)   # [16.0, ...]  (2^4)
```

---

## Reduction

Reduction compresses data along specified dimensions — turning many values into fewer (or one).

```python
a_vec.reduce(op, init_value, reduction_profile=...)
```

### Parameters

**`op`**: The combining function — `ReductionOp.ADD`, `MUL`, `MAX`, `MIN`.

**`init_value`**: The mathematical identity element for the operation:

| Operation | Correct init | Why |
|-----------|-------------|-----|
| ADD | 0.0 | 0 + x = x |
| MUL | 1.0 | 1 × x = x |
| MAX | -∞ | any x > -∞ |
| MIN | +∞ | any x < +∞ |

Choosing the wrong init pollutes results. The tutorial demonstrates this: using `init_value=1.0` with ADD adds an extra 1 to each reduced position.

**`reduction_profile`**: Controls which dimensions to collapse:

- `None` = **keep** this dimension
- `1` (or any non-None) = **reduce** (collapse) this dimension
- `0` = reduce **all** dimensions (full reduction to scalar)

### Worked Example

Matrix `[[1,2,3], [4,5,6]]`, shape `(2, 3)`:

```
        col0  col1  col2
row0  [  1,    2,    3  ]
row1  [  4,    5,    6  ]
```

**Full reduction** — `reduction_profile=0`:

```
1 + 2 + 3 + 4 + 5 + 6 = 21  (scalar)
```

**Reduce along dim 1 (columns collapse, rows survive)** — `reduction_profile=(None, 1)`:

Equivalent to `np.sum(a, axis=1)`.

```
row0: 1 + 2 + 3 = 6
row1: 4 + 5 + 6 = 15
→ [6, 15]    shape (2,3) → (2,)
```

**Reduce along dim 0 (rows collapse, columns survive)** — `reduction_profile=(1, None)` with `init_value=1.0`:

Equivalent to `np.sum(a, axis=0)`, but init=1.0 adds an extra 1:

```
col0: 1.0 + 1 + 4 = 6
col1: 1.0 + 2 + 5 = 8
col2: 1.0 + 3 + 6 = 10
→ [6, 8, 10]    shape (2,3) → (3,)
```

With `init_value=0.0`, the result would be `[5, 7, 9]` — the pure column sums.

**Mnemonic**: `None` = "I keep this axis"; non-None = "I crush this axis"; `0` = "crush everything."

### Why Reduction Matters on GPU

Nearly every important neural network operator depends on reduction:

- **Softmax**: MAX reduction (numerical stability) + ADD reduction (normalization denominator), both along the sequence dimension
- **LayerNorm**: ADD reduction to compute mean and variance per row
- **Loss functions**: Full ADD reduction to aggregate per-sample losses
- **FlashAttention**: Online softmax uses streaming MAX and ADD reductions
- **Pooling**: Global Average Pooling = ADD reduction over spatial dims; Max Pooling = MAX reduction

Performing these reductions at the register level (via TensorSSA) means no intermediate writes to shared or global memory — maximizing throughput.

---

## Broadcast

Broadcasting is the **inverse of reduction** — it expands data rather than compressing it.

The core problem it solves: you want to operate on two tensors of different shapes, but element-wise ops require matching shapes. Broadcasting automatically "replicates" the smaller tensor to align.

### Rules (same as NumPy)

1. **Pad dimensions**: If ranks differ, prepend 1s to the shorter shape. `(3,)` becomes `(1, 3)` when paired with `(4, 3)`.
2. **Match check**: Along each dimension, sizes must be equal OR one of them must be 1.
3. **Expand**: Dimensions of size 1 are conceptually replicated to match the other tensor.

### Example 1: Explicit `broadcast_to`

```python
a = shape (1, 3): [0, 1, 2]
a.broadcast_to((4, 3))

# Dim 0 expands from 1 → 4 (replicate the single row 4 times):
# [[0, 1, 2],
#  [0, 1, 2],
#  [0, 1, 2],
#  [0, 1, 2]]
```

### Example 2: Implicit broadcast during arithmetic

```python
a = shape (1, 3): [0, 1, 2]
c = shape (4, 1): [0, 1, 2, 3]

a + c → shape (4, 3)
```

Both tensors expand to `(4, 3)`:

```
a broadcast:          c broadcast:          a + c:
[[0, 1, 2],          [[0, 0, 0],          [[0, 1, 2],
 [0, 1, 2],    +      [1, 1, 1],    =      [1, 2, 3],
 [0, 1, 2],           [2, 2, 2],           [2, 3, 4],
 [0, 1, 2]]           [3, 3, 3]]           [3, 4, 5]]
```

### Why Broadcasting Matters on GPU

Registers are extremely scarce (~255 32-bit registers per thread). Broadcasting lets you use a single scalar or small vector in operations with large tensors without allocating registers for a full-sized copy of repeated values.

Real scenarios:

- **Softmax**: `row_max` has shape `(batch,)` but must be subtracted from shape `(batch, seq_len)` — broadcast along the column dimension
- **Bias addition**: FC output `(batch, hidden)` + bias `(hidden,)` — bias broadcasts along batch
- **Scaling**: Attention scores divided by `sqrt(d_k)` — scalar broadcasts to entire matrix
- **Normalization**: LayerNorm's mean/std `(batch, 1)` broadcast to `(batch, hidden)` for `(x - mean) / std`

### `make_rmem_tensor`

The broadcast examples in the tutorial use `cute.make_rmem_tensor((1, 3), dtype=...)` instead of loading from NumPy. This allocates a tensor directly in register memory — useful for constructing temporary intermediate values without round-tripping through host memory.

---

## Reduction and Broadcast: A Symmetric Pair

These two operations are conceptual inverses that almost always appear together in real kernels:

```
shape (4, 3)  ── reduce(axis=1) ──→  shape (4,)     compress: discard detail, keep summary
shape (4,)    ── broadcast_to    ──→  shape (4, 3)   expand: replicate summary to match original
```

### Softmax as a Case Study

```
logits (4, 3)
    │
    ├── reduce MAX along axis=1 → row_max (4,)         ← REDUCTION
    │       │
    │       └── broadcast to (4, 3)                     ← BROADCAST
    │               │
    ├── subtract → (logits - row_max) (4, 3)
    │
    ├── exp → exp_vals (4, 3)
    │
    ├── reduce ADD along axis=1 → row_sum (4,)          ← REDUCTION
    │       │
    │       └── broadcast to (4, 3)                     ← BROADCAST
    │               │
    └── divide → softmax_output (4, 3)
```

The "compress → expand" cycle is one of the fundamental patterns in GPU tensor computation.

---

## What TensorSSA Looks Like in a Real Kernel

### Vector Add — The Simplest Case

```python
@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    a_reg = a.load()          # LDG instruction: GMEM → register
    b_reg = b.load()          # LDG instruction: GMEM → register
    c_reg = a_reg + b_reg     # FADD instruction: register ← register + register
    c.store(c_reg)            # STG instruction: register → GMEM
```

This compiles roughly to:

```asm
LDG.E R0, [R4]       // a.load()
LDG.E R1, [R5]       // b.load()
FADD  R2, R0, R1     // a_reg + b_reg
STG.E [R6], R2       // c.store()
```

**TensorSSA objects correspond to R0, R1, R2 — actual hardware register contents.**

### Softmax — A Realistic Kernel

```python
@cute.kernel
def softmax_kernel(input: cute.Tensor, output: cute.Tensor):
    x = input.load()                                              # TensorSSA
    row_max = x.reduce(cute.ReductionOp.MAX, float('-inf'),
                       reduction_profile=0)                       # TensorSSA (scalar)
    x_shifted = x - row_max                                       # TensorSSA (broadcast + sub)
    x_exp = cute.math.exp(x_shifted)                              # TensorSSA (element-wise exp)
    exp_sum = x_exp.reduce(cute.ReductionOp.ADD, 0.0,
                           reduction_profile=0)                   # TensorSSA (scalar)
    result = x_exp / exp_sum                                      # TensorSSA (broadcast + div)
    output.store(result)                                          # write back to memory
```

Every intermediate — `x`, `row_max`, `x_shifted`, `x_exp`, `exp_sum`, `result` — is a TensorSSA living in registers. No data touches memory between load and store. That is why it is fast.

### Contrast: Raw CUDA C++ Without This Abstraction

```cpp
__global__ void softmax_kernel(float* input, float* output, int N) {
    float vals[4];
    vals[0] = input[tid * 4 + 0];   // manual load, element by element
    vals[1] = input[tid * 4 + 1];
    vals[2] = input[tid * 4 + 2];
    vals[3] = input[tid * 4 + 3];

    float max_val = vals[0];          // manual reduction loop
    max_val = fmaxf(max_val, vals[1]);
    max_val = fmaxf(max_val, vals[2]);
    max_val = fmaxf(max_val, vals[3]);

    float exp_vals[4];
    exp_vals[0] = expf(vals[0] - max_val);  // manual element-wise ops
    exp_vals[1] = expf(vals[1] - max_val);
    // ... repeat for every element ...

    float sum = exp_vals[0] + exp_vals[1] + exp_vals[2] + exp_vals[3];

    output[tid * 4 + 0] = exp_vals[0] / sum;
    // ... repeat ...
}
```

TensorSSA replaces all of this manual register management with `x.reduce(MAX, ...)` and `x - row_max`. It is a high-level abstraction over manual register-level programming.

---

## Summary

```
cute.Tensor  =  pointer + layout descriptor in memory (cannot compute)
                    │
                 .load()  =  hardware memory-read instruction
                    │
               TensorSSA  =  Python proxy for data living in GPU registers
                    │           • corresponds to an SSA value in MLIR IR
                    │           • immutable: each operation produces a new TensorSSA
                    │           • overloads Python operators → compiles to vectorized GPU instructions
                    │           • supports: arithmetic, comparison, bitwise, unary math,
                    │                       slicing, reduction, broadcast
                    │
                 .store()  =  hardware memory-write instruction
                    │
              cute.Tensor  =  result written back to memory
```

TensorSSA is not a "new format." It is a **Python-level handle to actual data in GPU registers**. `.load()` is not a "format conversion" — it is a real hardware data transfer from slow memory to fast registers. This step is necessary because the GPU's ALU can only operate on register contents. That is a hardware constraint, not a software design choice.
