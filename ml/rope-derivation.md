# Understanding RoPE: A Deep Dive Through Q&A

## Q1: What's the goal exactly?

Original Attention computes `q · k`. We want the inner product to **automatically encode the distance** between two tokens.

Formally, find a transformation `f` such that:
- `q` at position `m` → `f(q, m)`
- `k` at position `n` → `f(k, n)`
- `⟨f(q,m), f(k,n)⟩` depends **only on `m−n`**, not on `m` or `n` individually

**Why?** Attention should care about "how far apart we are," not "I'm token #3, you're token #8." The absolute starting point of a sentence shouldn't matter.

---

## Q2: Why rotation?

**Key geometric insight:** Rotation naturally preserves only relative angles.

Picture two clock hands:
- `q` points one direction, `k` points another
- Their inner product = `‖q‖ · ‖k‖ · cos(angle between them)`

If you rotate `q` by `m` degrees and `k` by `n` degrees:
- Lengths unchanged
- New angle between them = `(original angle) + (m − n)`
- **Only the difference `m − n` shows up**

### Concrete check
- `q` at 0°, `k` at 30°. Original gap: 30°
- Rotate by `m=5, n=2`: new positions 5°, 32°. Gap: **27°**
- Rotate by `m=100, n=97`: new positions 100°, 127°. Gap: **27°**

Different absolute positions, same difference (`m−n = −3`), same gap → same inner product. ✓

---

## Q3: How is the rotation formula derived?

Starting from the constraint `⟨f(q,m), f(k,n)⟩ = g(q,k, m−n)`:

### Step 1: Switch to complex numbers
2D vectors ↔ complex numbers. Two useful facts:
- Inner product: `⟨q, k⟩ = Re[q · k̄]`
- Polar form: `z = R · e^(iθ)` makes multiplication = "multiply magnitudes, add angles"

### Step 2: Decompose into magnitude and phase
Write `f(q,m) = R_f(q,m) · e^(iΘ_f(q,m))`. The constraint splits into two independent equations:

- **Magnitude equation:** `R_f(q,m) · R_f(k,n) = R_g(q,k, m−n)`
- **Phase equation:** `Θ_f(q,m) − Θ_f(k,n) = Θ_g(q,k, m−n)`

### Step 3: Force magnitude to stay constant
Plug in `m = n = 0` with initial condition `f(q,0) = q`:

`R_f(q,m) · R_f(k,m) = ‖q‖ · ‖k‖`

Right side has no `m` → `R_f` doesn't depend on `m`. Set `R_f(q,m) = ‖q‖`.

**Conclusion: `f` cannot stretch — only rotate.**

### Step 4: Force phase to grow linearly
- Plug in `m = n` → `Θ_f(q,m) = Θ(q) + φ(m)`, where `φ` depends only on `m`
- Plug in `n = m−1` → `φ(m) − φ(m−1) = constant`

Call that constant `θ`. Then `φ(m) = m · θ` (arithmetic sequence, with `φ(0) = 0`).

**Where `θ` comes from:** It's not chosen — it falls out of the derivation as the forced step size between consecutive positions.

### Step 5: Assemble
`f(q,m) = ‖q‖ · e^(i(Θ(q) + mθ)) = q · e^(imθ)`

Translate complex multiplication into matrix form:

```
f(q, m) = | cos(mθ)  −sin(mθ) | | q₀ |
          | sin(mθ)   cos(mθ) | | q₁ |
```

This is the standard 2D rotation matrix.

---

## Q4: Does the final matrix actually satisfy the original constraint?

Yes — verify directly.

### Two key properties of rotation matrices
1. **Transpose = reverse rotation:** `R_m^⊤ = R_{−m}`
2. **Composition = adding angles:** `R_a · R_b = R_{a+b}`

### Plug into the inner product
```
⟨R_m q, R_n k⟩ = q^⊤ R_m^⊤ R_n k
              = q^⊤ R_{−m} R_n k
              = q^⊤ R_{n−m} k
```

**Only `n−m` survives — `m` and `n` individually disappear.** ✓

### Numerical sanity check
Let `q = (1,0)`, `k = (0,1)`, `θ = π/2`.

| Case | `m` | `n` | `m−n` | Inner product |
|------|-----|-----|-------|---------------|
| A    | 1   | 3   | −2    | 0             |
| B    | 5   | 7   | −2    | 0             |
| C    | 0   | 1   | −1    | −1            |

Same `m−n` → same result, regardless of absolute positions.

---

## The full causal chain

```
Wish: inner product depends only on relative distance
            ⟨f(q,m), f(k,n)⟩ = g(q,k, m−n)
                       ↓
            (rigorous derivation via complex numbers)
                       ↓
Conclusion: f must be a rotation
            f(q,m) = R_m · q
                       ↓
            (substitute back to verify)
                       ↓
Confirmed: ⟨R_m q, R_n k⟩ = q^⊤ R_{n−m} k
                       ↓
            Loop closed ✓
```

The constraint says **"this is what I want."** The matrix says **"this is what it must be."** The two ends meet — that's why RoPE is theoretically clean: it's the *exact solution* to a precise constraint, not a heuristic approximation like Sinusoidal encoding.

---

## Reference

- [让研究人员绞尽脑汁的Rotary Position Embedding (RoPE)](https://spaces.ac.cn/archives/8265) — 苏剑林（苏神）博客，原始推导来源
