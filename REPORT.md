# sld-poc — Findings & Roadmap

*A proof-of-concept **spectral continuous-latent diffusion language model**, built from scratch in JAX on TPU (Colab v5e), at TinyStories scale.*

---

## 1. The idea

Standard language models emit one discrete token at a time. This PoC follows the **CALM** direction (continuous next-vector prediction) and pushes it toward **diffusion**:

1. A **codec** compresses every **K = 4 tokens → one continuous latent vector**.
2. A **diffusion denoiser** generates a whole sequence of those vectors **in parallel**, by iteratively denoising from Gaussian noise — instead of left-to-right token generation.
3. A **spectral variant** diffuses the latents in the **frequency domain** (an orthonormal DCT across positions), so generation proceeds coarse → fine.

Continuous latents are a natural fit for Gaussian diffusion, and the K-fold compression means the diffuser works over `L/K` positions instead of `L` tokens — two compounding efficiency axes, plus abilities autoregression lacks (infilling, self-correction).

---

## 2. What we built

| Stage | Module | What it does | Status |
|------|--------|--------------|--------|
| **1** | `src/codec.py` | VAE codec: K=4 tokens ↔ 1 latent (64-d), KL-regularized, noise-robust | ✅ |
| **2** | `src/diffusion.py` | Self-conditioned latent-diffusion denoiser; generate / infill / self-correct | ✅ |
| **3** | `src/spectral.py` | Frequency-domain (DCT) variant + baseline-vs-spectral comparison | ✅ |

Everything runs in JAX/Flax on a single TPU core, with Drive-backed checkpoint/resume. Reproduction notebooks are in `notebooks/`.

---

## 3. Results

### Codec (Stage 1) — strong
- Config: K = 4 tokens → 64-d latent (`D_EMB=128, D_HIDDEN=512`), VAE with KL + σ≈0.3 noise injection.
- **~99.96% per-token reconstruction** after 20k steps. Reconstruction loss → ~0.001; KL fell from ~16 → ~2.6 over training, i.e. the latent space became *smoother* (more diffusion-friendly) as it got *more* accurate.
- Round-trip of held-out text is near-perfect.

### Diffusion + Spectral (Stages 2–3) — mechanism works, quality is the gap
- Denoiser: bidirectional transformer (`DIM=512, DEPTH=8`) over M=16 latent positions (= 64 tokens of context), self-conditioned, x0-prediction, cosine schedule, 80k steps.
- Training is stable; MSE on standardized latents plateaus around ~0.42 (noisy, as expected for averaged-over-noise-levels diffusion loss).
- **Generation is locally plausible but globally incoherent** — real TinyStories vocabulary (`Lily`, `Jack`, `"once upon a time"`, `forest`), correct local fragments, but no story-level coherence (word-salad).
- **Spectral vs baseline:** the spectral variant shows **no clear coherence advantage** at this scale — consistent with the hypothesis that a frequency basis is a *representational* aid, not a *capacity* fix.
- **Parallel decode:** 50 denoising passes produce a 64-token sequence vs 64 sequential passes for autoregression — the structural speedup is real. (Current tok/s is measured under an *eager* sampler and is a lower bound; the jitted path is much faster.)

---

## 4. Key finding

> **The codec and the full "generate-by-denoising" mechanism work end-to-end. The bottleneck is denoiser quality at aggressive K = 4 compression** — the known hard frontier of continuous-latent *text* diffusion.

The latent text manifold is unforgiving: small errors in a 64-d vector decode to the *wrong tokens* (text decode is a sharp function, unlike images where a blurry latent is just a blurry pixel). A modest denoiser trained on a small budget can't model that manifold precisely enough, and — importantly — **changing the basis (spectral) doesn't add the missing capacity.** Coherent generation needs *scale*, not a different transform.

This is a clean, honest result: the architecture is sound; the open problem is precisely characterized.

---

## 5. Roadmap

### 5.1 Scale the denoiser *(the direct fix)*
Bigger model, more steps, longer context (M), and few-step distillation. This is the most likely path from word-salad to coherent text, and the primary use of additional compute.

### 5.2 AST-aware chunking *(for code)*
Don't waste latent capacity on syntax a parser already knows. Re-render the deterministic skeleton **by rule** and encode only the content holes:

```
fn <1> ( ) -> <2> { <3> <4> }
   ▲name      ▲ret   ▲body…
```

A function becomes a **handful of content-vectors** (`<1>…<4>`) instead of the ~15–20 tokens it normally costs — and because `fn`, `(`, `)`, `->`, `{`, `}` are printed by rule, **the output is syntactically valid by construction** and that fraction of tokens is *lossless*. (Tree-sitter / `syn` provides the structure at data-prep time only; it never touches inference.)

### 5.3 Information-adaptive (variable-K) chunking
Allocate vectors by **content**, not token count, so each vector carries ~constant information:
- Low-entropy, predictable spans — e.g. **"once upon a time" → 1 vector**.
- High-entropy spans (rare names, numbers, identifiers) → their own **short** chunks.

This rate-matches the payload to the codec's fixed capacity, removing the worst case (a dense 4-token chunk overflowing one vector) and keeping reconstruction uniformly high — directly attacking the quality gap from §4.

### 5.4 Decoder-aware training & self-correction at scale
Train the denoiser with a term that pushes it toward *decodable* latents (not just MSE-close ones), and use the codec's own reconstruction/prior signal to flag and re-denoise suspect chunks — the self-correction loop diffusion uniquely enables.

---

## 6. Why this needs TPUs

Each item in §5 — scaling the denoiser, plus learning structural/adaptive chunkers and retraining the codec around them — is compute-bound. The PoC validates the pipeline end-to-end on a single TPU core and pinpoints the bottleneck; closing it requires substantial parallel compute. That is the work we are requesting TPU access for.

---

## 7. Reproduce
- `notebooks/01_codec.ipynb` — train the codec
- `notebooks/02_diffusion.ipynb` — latent diffusion + demos
- `notebooks/03_spectral.ipynb` — spectral variant + comparison

Each notebook clones this repo fresh and runs in-kernel on a Colab TPU runtime.
