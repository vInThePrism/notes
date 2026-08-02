# CuTe DSL – Hello World: Technical Learning Notes

## What Is CuTe DSL?

CuTe DSL is a domain-specific language embedded in Python. You write programs using Python syntax, but CuTe compiles the supported portions into CPU launch code and GPU kernel code. **The Python interpreter does not run directly on the GPU.**

Official introduction: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html>

## Core Concepts

| Concept | Meaning |
|---------|---------|
| **Host** | The CPU side |
| **Device** | The GPU side (NVIDIA GPU) |
| **Kernel** | A function executed on the GPU |
| **Thread** | A single GPU worker |
| **Block / CTA** | A group of threads that can cooperate |
| **Grid** | All the blocks launched by a single kernel invocation |
| **Warp** | The basic scheduling unit on NVIDIA GPUs — 32 threads |
| **JIT** | Just-In-Time compilation — code is compiled right before it is needed |

## The Two Kinds of Functions

GPU programming requires **two separate roles**:

1. **CPU function (`@cute.jit`)** — runs on the host. Responsible for preparing data, configuring the launch grid, and submitting the kernel to the GPU.
2. **GPU function (`@cute.kernel`)** — defines what every GPU thread does once launched. All threads execute the same code; each thread identifies its own portion of work through its thread index.

The CPU does **not** tell each GPU thread what to do one by one. Instead, the CPU issues a single launch command ("start N blocks × M threads"), and every thread runs the same kernel, using its own index to decide which data to process.

```
CPU function (@cute.jit)
    ↓  decides grid/block dimensions
    ↓  launches kernel
GPU function (@cute.kernel)
    ↓  all threads execute the same code
    ↓  each thread uses its own index to pick its work
```

## Imports

```python
import cutlass
import cutlass.cute as cute
```

- `cutlass` — top-level CUTLASS module; provides data types, compilation helpers, and CUDA utilities.
- `cutlass.cute` — main CuTe DSL interface: decorators, GPU architecture operations, kernel launch API, etc.

Importing alone does not execute any GPU program.

## Writing the GPU Kernel

```python
@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello world\n")
```

### `@cute.kernel`

Marks the function to be compiled into a GPU kernel. Defining it does **not** run it; it must later be launched via `.launch()`.

### `cute.arch.thread_idx()`

Returns a 3D thread index `(x, y, z)` **local to the current block** (not a global index across the entire grid).

- `tidx` receives the x component.
- `_` discards y and z (unused here).

With `block=(32, 1, 1)`, `tidx` ranges from 0 to 31 within this block.

API reference: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_arch.html>

### Why Only Thread 0 Prints

All 32 threads execute the kernel. Only the thread whose `tidx == 0` enters the `if` branch and calls `cute.printf`. Without the guard, all 32 threads would print.

**Important:** `thread_idx()` resets to 0 in every block. If you launched 10 blocks and only checked `tidx == 0`, you would get 10 prints — one per block. Here there is only 1 block, so exactly 1 print.

### `dynamic_expr` (Deprecated)

The original tutorial used:

```python
if cutlass.dynamic_expr(tidx == 0):
```

This explicitly told the old CuTe compiler that the condition must be evaluated at runtime. Since CUTLASS 4.1, `dynamic_expr` is deprecated. A plain `if` is now compiled as a runtime branch by default; only `cutlass.const_expr(...)` forces a compile-time decision.

References:
- Control flow: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html>
- Changelog: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html>

## Writing the CPU Launch Function

```python
@cute.jit
def hello_world():
    cute.printf("hello world")          # host-side print (lowercase h)

    kernel().launch(
        grid=(1, 1, 1),                 # 1 block
        block=(32, 1, 1),               # 32 threads per block
    )
```

### `@cute.jit`

Defines a host-side JIT-compiled function. On first use, CuTe analyzes and compiles it into an executable host entry point plus any required GPU modules.

### `cute.printf` Inside a `@cute.jit` Function

This `cute.printf` is compiled into the host-side runtime code and produces CPU output. The tutorial intentionally uses lowercase `"hello world"` (CPU) vs. uppercase `"Hello world"` (GPU) to distinguish the two sources.

**Three kinds of print to distinguish:**

| Where | What | When It Runs |
|-------|------|--------------|
| Outside any decorator | `print(...)` | Normal Python execution |
| Inside `@cute.jit` | `print(...)` | Typically at DSL compile time |
| Inside `@cute.jit` or `@cute.kernel` | `cute.printf(...)` | Compiled into the generated runtime code |

### Kernel Launch Parameters

```python
kernel().launch(
    grid=(1, 1, 1),     # 1 × 1 × 1 = 1 block
    block=(32, 1, 1),   # 32 × 1 × 1 = 32 threads per block
)
```

Think of it as: **prepare kernel → specify execution dimensions → submit to GPU**.

- **Grid** is measured in blocks. `(1,1,1)` = 1 block total.
- **Block** is measured in threads. `(32,1,1)` = 32 threads.
- Total threads = 1 block × 32 threads = **32 threads** (exactly one warp).

Only thread 0 prints, but all 32 threads are launched; the other 31 evaluate the condition as false and exit.

Parameters not provided (stream, dynamic shared memory size) use their defaults.

## Initializing the CUDA Context

```python
cutlass.cuda.initialize_cuda_context()
```

A CUDA context is the runtime environment linking the current process to a specific GPU. GPU modules, kernels, memory operations, and execution state all belong to a context.

The tutorial makes initialization explicit to:

- Detect CUDA errors early.
- Control when the context is created.
- Avoid conflicts when multiple libraries each try to create their own context.

## Two Ways to Run the Program

### Method 1 — JIT (Compile and Run Immediately)

```python
hello_world()
```

On first call in a fresh environment:

1. Compile `hello_world` (and the GPU kernel it references).
2. Execute the CPU launch function.
3. Submit the kernel to the GPU.

CuTe caches JIT results, so subsequent calls may reuse the compiled artifacts.

### Method 2 — Explicit Compile, Then Run

```python
hello_world_compiled = cute.compile(hello_world)   # compile only
hello_world_compiled()                              # execute later
```

`cute.compile(...)` returns a callable **JIT Executor** without executing anything. Calling it later runs the already-compiled host function and launches the kernel. This separates compilation overhead from execution — useful for repeated runs.

Note: this is still JIT workflow (compile is triggered by calling `cute.compile`), not full ahead-of-time (AOT) compilation that produces a standalone binary.

Reference: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.html>

## Keeping PTX and CUBIN

```python
from cutlass.cute import KeepPTX, KeepCUBIN

hello_world_compiled_ptx_on = cute.compile[KeepPTX, KeepCUBIN](hello_world)
```

This recompiles and retains the generated GPU artifacts:

- **PTX** — NVIDIA's virtual GPU instruction text; an intermediate representation similar to GPU assembly.
- **CUBIN** — binary code assembled for a specific GPU architecture; the actual machine instructions the GPU executes.

These files are useful for inspecting compiler output, performance analysis, and debugging. Keeping them does not change computation results.

Note: the tutorial creates `hello_world_compiled_ptx_on` but **never calls it**, so this version is only compiled, not executed.

Reference: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html>

## Understanding the Output Order

```
Running hello_world()...
Compiling...
hello world
Hello world
Compiling with PTX/CUBIN dumped...
Running compiled version...
hello world
Hello world
```

| Output | Source |
|--------|--------|
| `Running hello_world()...` | Python `print` |
| `Compiling...` | Python `print` |
| `hello world` | Host function's `cute.printf` (first execution) |
| `Hello world` | GPU kernel's `cute.printf` (first execution) |
| `Compiling with PTX/CUBIN dumped...` | Python `print` |
| `Running compiled version...` | Python `print` |
| `hello world` | Pre-compiled host function (second execution) |
| `Hello world` | GPU kernel (second execution) |

### Why Does "Compiling..." Appear Before the First Kernel Output?

GPU kernel launches are **asynchronous**: the CPU submits work to the GPU and immediately continues executing subsequent Python statements. The GPU completes its work (and flushes its printf buffer) later. Additionally, `cute.printf` without `\n` and mixed GPU/host output may involve separate buffering. The exact interleaving is **not guaranteed**; to enforce ordering, you need explicit synchronization.

## Key Takeaways

1. `@cute.jit` defines a **CPU-side** function for compilation, configuration, and kernel launch.
2. `@cute.kernel` defines the function that **GPU threads** actually execute.
3. `grid=(1,1,1)`, `block=(32,1,1)` means 1 block with 32 threads.
4. All 32 threads enter the kernel, but `if tidx == 0` restricts the print to a single thread.
5. The CPU does not micro-manage each thread; it writes **one shared program** and each thread picks its own work based on its index.
