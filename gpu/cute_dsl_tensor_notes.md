# CuTe DSL: Tensors, GPU Execution Model & Thread Hierarchy

## 1. The Core Problem: Memory Is Always Flat

GPU memory (VRAM) is physically a one-dimensional, contiguous block of bytes. A 8×5 matrix of float32 occupies 160 bytes laid out linearly:

```
Address: [0][1][2][3][4][5][6][7][8][9]...[39]
```

To access "row 2, column 3", you must manually compute the offset: `2 * 5 + 3 = 13`. This is manageable for simple cases, but becomes a nightmare when tiling, transposing, or moving data between memory levels.

**CuTe Tensor solves this by attaching a coordinate system to a block of memory**, so you can read/write using `[row, col]` instead of computing offsets by hand.

---

## 2. Tensor = Engine ∘ Layout

A CuTe Tensor is the composition of two parts:

### Engine

A pointer to the start of a memory region. It supports two operations:
- **Offset**: `engine + n` → move n elements forward
- **Dereference**: `*engine` → read/write the value at current position

### Layout

A mapping function from coordinates to offsets, defined by **shape** and **stride**:
- **Shape**: the dimensions, e.g. `(8, 5)` means 8 rows × 5 columns
- **Stride**: how many elements to jump per step in each dimension

Example: `stride=(5, 1)` means moving one row jumps 5 elements in memory, moving one column jumps 1 element. This is **row-major** order.

### Evaluation Formula

```
T(c) = *(Engine + Layout(c))
```

When you write `tensor[2, 3]`:
1. Layout computes offset: `2*5 + 3*1 = 13`
2. Engine jumps to `start_address + 13`
3. Dereference reads the value

### Changing Interpretation Without Copying Data

The same 40-element memory block can be read as different shapes by changing the Layout:
- `shape=(8,5), stride=(5,1)` → 8×5 row-major matrix
- `shape=(5,8), stride=(1,5)` → 5×8 matrix (effectively a transpose)

No data is moved. Only the "interpretation rules" change.

---

## 3. Creating Tensors in CuTe DSL

### Method 1: Pointer + Layout (Manual)

```python
@cute.jit
def create_tensor_from_ptr(ptr: cute.Pointer):
    layout = cute.make_layout((8, 5), stride=(5, 1))
    tensor = cute.make_tensor(ptr, layout)
    tensor.fill(1)
    cute.print_tensor(tensor)
```

### Method 2: DLPack Protocol (From PyTorch / NumPy / JAX)

```python
from cutlass.cute.runtime import from_dlpack

a = torch.randn(8, 5, dtype=torch.float32, device='cuda')
cute_tensor = from_dlpack(a)  # extracts pointer + layout, no data copy
```

`from_dlpack` extracts the existing memory address (Engine) and shape/stride info (Layout) from the framework tensor and wraps them into a CuTe Tensor. The data stays in the same memory location.

---

## 4. Tensor Access Methods

### Full Evaluation — returns a scalar

```python
a[2, 4]        # read value at row 2, col 4
a[2, 3] = 100  # write value
```

### Linear Index — auto-mapped to coordinates

```python
a[9]  # CuTe uses column-major (lexicographic) ordering to convert to (row, col)
```

### Partial Evaluation (Slicing) — returns a sub-tensor

Providing incomplete coordinates returns a lower-dimensional tensor. The known coordinates get "absorbed" into the Engine offset; the remaining coordinates form a new Layout.

```
T(c) = (E + L(c')) ∘ L(c*) = T'(c*)
```

---

## 5. Memory Spaces

CuTe Tensors can be bound to different CUDA memory spaces:

| Memory Space | Scope | Latency | Capacity |
|---|---|---|---|
| **Global (gmem)** | All threads across all blocks | High | Large (GBs) |
| **Shared (smem)** | All threads within one block | Low | Limited (~48-228 KB) |
| **Register (rmem)** | Single thread only | Lowest | Very small |
| **Tensor (tmem)** | Blackwell architecture only | — | — |

The high-performance GPU programming pattern: global memory → shared memory → registers → compute → write back. CuTe Tensor provides a uniform `tensor[coord]` interface at every level.

---

## 6. Identity Tensor and Coordinate Mapping

### The Problem

GPU threads get a linear ID (0, 1, 2, ..., 39), but data is multi-dimensional (8×5 matrix). You need a way to convert `thread_id → (row, col)`.

### Identity Tensor

An identity tensor satisfies `I(c) = c` — input a coordinate, get the same coordinate back. Used as a lookup table:

```python
coord_tensor = cute.make_identity_tensor(shape)
coord = coord_tensor[thread_id]  # linear index → (row, col)
```

For shape `(8, 5)`, CuTe uses column-major lexicographic ordering:

```
index 0  → (0, 0)
index 1  → (1, 0)
...
index 7  → (7, 0)
index 8  → (0, 1)
index 9  → (1, 1)
...
index 13 → (5, 1)
...
index 39 → (7, 4)
```

Formula: `idx = c₁ + c₂·s₁ + c₃·s₁·s₂ + ...`

---

## 7. GPU Execution Model

### Every Thread Runs the Same Code

A `@cute.kernel` function is not executed by you — it's a set of instructions handed to the GPU. Every thread runs the **exact same code**; the only difference is each thread's built-in ID (`threadIdx.x`, `blockIdx.x`).

Like handing the same instruction sheet to 40 workers: "Check your worker number, go to the corresponding station, do the work there."

### Full Example: Element-wise Addition

```python
@cute.kernel
def add_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    # Step 1: Who am I?
    tid = cute.arch.threadIdx.x

    # Step 2: What position am I responsible for?
    coord = cute.make_identity_tensor(A.layout.shape)[tid]

    # Step 3: Read, compute, write
    C[coord] = A[coord] + B[coord]


@cute.jit
def launch_add(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    add_kernel[1, 40](A, B, C)  # 1 block, 40 threads


A = torch.randn(8, 5, dtype=torch.float32, device='cuda')
B = torch.randn(8, 5, dtype=torch.float32, device='cuda')
C = torch.zeros(8, 5, dtype=torch.float32, device='cuda')

launch_add(from_dlpack(A), from_dlpack(B), from_dlpack(C))
```

### What Thread 13 Does

```
tid = 13
coord = identity_tensor[13] → (5, 1)

A[5, 1]:  Layout computes offset: 5*5 + 1*1 = 26
          Engine: A_start_address + 26*4bytes → read 3.14

B[5, 1]:  same offset calculation → read 2.0

C[5, 1]:  same offset → write 5.14
```

All 40 threads execute simultaneously, each handling one element.

---

## 8. Thread Hierarchy

GPU threads are organized in layers:

```
Grid
 └── Block (CTA)          ← you specify how many
      └── Warp             ← hardware auto-groups every 32 threads
           └── Thread      ← smallest unit
```

### Thread

The smallest execution unit. Each runs the kernel code with its own ID.

### Warp (32 threads)

A **hardware-level** execution group. 32 consecutive threads execute the **same instruction** at the **same clock cycle**, each operating on different data (SIMT model). You don't create or specify warps — hardware does it automatically.

### Block / CTA (Cooperative Thread Array)

You specify this at launch. Key properties:
- Threads in the same block can **share shared memory**
- Threads in the same block can **synchronize** (barrier)
- Cross-block communication is **not** directly possible
- One block runs on one SM (Streaming Multiprocessor)
- Max threads per block: typically 1024

### Grid

The collection of all blocks for one kernel launch. Spans the entire GPU.

### Launch Configuration

```python
kernel[num_blocks, threads_per_block](args)

kernel[1, 40]     # 1 block × 40 threads = 40 total
kernel[4, 256]    # 4 blocks × 256 threads = 1024 total
kernel[100, 128]  # 100 blocks × 128 threads = 12800 total
```

### What You Control vs What Hardware Does

| You decide | Hardware does automatically |
|---|---|
| Number of blocks in the grid | Group every 32 threads into a warp |
| Number of threads per block | Schedule blocks onto SMs |

### Why Multiple Blocks?

- One block is limited to ~1024 threads
- One block can only run on one SM
- GPU has many SMs — multiple blocks run in parallel on different SMs
- This is how you fully utilize the GPU

### Warp Sizing Tip

You can launch any number of threads per block (even 1 or 7). But hardware always allocates full warps of 32. Launching 7 threads means 25 slots are wasted. In practice, always use multiples of 32 (128, 256, 512) to avoid wasting hardware resources.

---

## 9. How Memory Allocation Works

When you write `torch.randn(8, 5, device='cuda')`, PyTorch requests 160 bytes from GPU VRAM:

```
GPU VRAM (e.g. 8 GB total)
┌─────────────────────────────┐
│  ... system / other usage   │
├─────────────────────────────┤
│  A: 40 floats (160 bytes)   │  ← A.data_ptr() = 0x1000
├─────────────────────────────┤
│  B: 40 floats (160 bytes)   │  ← B.data_ptr() = 0x10A0
├─────────────────────────────┤
│  C: 40 floats (160 bytes)   │  ← C.data_ptr() = 0x1140
├─────────────────────────────┤
│  ... free space             │
└─────────────────────────────┘
```

- `A.data_ptr()` returns the starting address — this becomes the **Engine**
- Shape and stride info becomes the **Layout**
- `from_dlpack(A)` packages both into a CuTe Tensor (no data copy)

Three tensors A, B, C point to three different regions. The kernel reads A and B, writes C. Addresses never overlap, so threads don't conflict.

---

## 10. Key Takeaways

1. **Tensor = Engine (pointer) + Layout (coordinate-to-offset mapping)**
2. **Layout lets you reinterpret the same memory as different shapes without copying**
3. **`@cute.kernel` is code for threads; `@cute.jit` is code for the CPU/host**
4. **All threads run the same kernel; `threadIdx.x` differentiates what each thread does**
5. **Identity tensor converts linear thread IDs to multi-dimensional coordinates**
6. **Thread hierarchy: Grid → Block → Warp → Thread; you control Grid and Block sizes**
7. **CuTe Tensor provides a uniform `tensor[coord]` interface across all memory spaces**
