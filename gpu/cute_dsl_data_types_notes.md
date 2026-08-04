# CuTe DSL Data Types & The Compile-Time vs Runtime Mental Model

## The Core Concept: Where Does Your Code Actually Run?

A `@cute.jit` function goes through two phases:

1. **Compile time (CPU)**: Python interpreter reads your code and translates it into GPU machine code (PTX/CUBIN).
2. **Runtime (GPU)**: The compiled machine code is sent to the GPU and actually executed.

This split is the root of everything below.

## Compile-Time vs Runtime Values

### The Problem

`print()` is a Python function — it runs at **compile time**, before code is sent to the GPU. At that point, CUTLASS typed values don't exist yet; they're just placeholders telling the compiler "a value of this type will be here later." So `print()` outputs `?`.

`cute.printf()` gets compiled **into** the GPU code and executes at **runtime**, when the values are real. So it prints the actual numbers.

```python
@cute.jit
def foo():
    a = cutlass.Float32(3.14)
    print(a)              # Output: ?         (compile time — value doesn't exist yet)
    cute.printf("{}", a)  # Output: 3.140000  (runtime — GPU has the real value)
```

### Analogy

Think of writing a recipe (code) and handing it to a chef (GPU).

- `print()` = asking "does this taste good?" **while writing the recipe**. The dish doesn't exist yet.
- `cute.printf()` = writing in the recipe: "taste and report after this step." The chef does it when cooking.

### Why This Matters

This is not theoretical — it's a practical debugging issue. If you use `print()` to inspect variables in GPU code, you'll see nothing but `?`. You **must** use `cute.printf()` to see what's happening on the GPU.

## Python Literals vs CUTLASS Types

```python
@cute.jit
def foo():
    a = 3.14                    # Plain Python float — baked into GPU code as a fixed constant
    b = cutlass.Float32(3.14)   # CUTLASS type — a runtime variable on the GPU
```

- `a = 3.14` is a **compile-time constant**. The compiler hardcodes `3.14` into the GPU binary. It can never change.
- `b = cutlass.Float32(3.14)` is a **runtime variable**. It can be modified, participate in computations, and hold values that aren't known at compile time (e.g., data read from input, loop accumulators).

**Bottom line**: `cutlass.Float32` is a variable for the GPU. `3.14` is a constant for the compiler.

## CUTLASS Numeric Type System

### Integer Types

| Type | Description |
|------|-------------|
| `Int8` / `Uint8` | 8-bit signed / unsigned |
| `Int16` / `Uint16` | 16-bit signed / unsigned |
| `Int32` / `Uint32` | 32-bit signed / unsigned |
| `Int64` / `Uint64` | 64-bit signed / unsigned |
| `Int128` / `Uint128` | 128-bit signed / unsigned |

### Floating Point Types

| Type | Description |
|------|-------------|
| `Float16` | 16-bit float |
| `Float32` | 32-bit float (standard) |
| `Float64` | 64-bit float (double) |
| `BFloat16` | Brain Float 16-bit (used in ML training) |
| `TFloat32` | Tensor Float 32 (reduced precision for tensor ops) |
| `Float8E4M3` | 8-bit float, 4-bit exponent, 3-bit mantissa |
| `Float8E5M2` | 8-bit float, 5-bit exponent, 2-bit mantissa |

## Type Conversion with `.to()`

Convert between types using the `.to()` method:

```python
x = cutlass.Int32(42)
y = x.to(cutlass.Float32)   # 42 → 42.0
```

### Gotchas

- **Float → Int truncates**: `Float32(3.14).to(Int32)` → `3` (decimal part dropped)
- **Overflow wraps/clamps**: `Int32(300).to(Int8)` → `44` (300 exceeds Int8 range of -128 to 127)

## Operator Overloading

CUTLASS types support standard Python operators directly:

- **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- **Comparison**: `<`, `<=`, `==`, `!=`, `>=`, `>`
- **Bitwise**: `&`, `|`, `^`, `<<`, `>>`
- **Unary**: `-` (negate), `~` (bitwise NOT)

### Mixed-Type Promotion

When you mix types, the narrower type gets promoted (like C's implicit conversion):

```python
a = cutlass.Int32(10)
x = cutlass.Float32(5.5)
result = a + x  # Int32 promoted to Float32 → result is Float32(15.5)
```

Also works with plain Python numbers:

```python
y = x * 2  # Python int 2 is handled automatically → Float32(11.0)
```
