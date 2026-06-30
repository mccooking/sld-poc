# sld-poc

A **proof-of-concept** continuous-latent **diffusion language model**, with a **spectral** (frequency-domain) variant — built in JAX/Flax for **TPU (v5e)**.

## Idea
Compress every *K* tokens into one continuous vector (a CALM-style codec), then **generate text by denoising a grid of those vectors in parallel** — diffusion, instead of one token at a time. The spectral variant diffuses the latents in the **frequency domain** (a DCT across positions), so generation proceeds coarse → fine for better global structure.

Capabilities it demonstrates that autoregressive models can't: **parallel generation, infilling, and self-correction.**

> Scope: a deliberately small PoC (TinyStories scale) to validate the approach end-to-end and characterize the compression-vs-quality frontier. The full-scale follow-up lives in a separate repo.

**📄 Findings & roadmap: [`REPORT.md`](REPORT.md)** — what worked, what didn't, and where it goes next.

### TL;DR of results
- **Codec works** — K=4 tokens ↔ 1 latent at **~99.96%** reconstruction.
- **Mechanism works** — generate-by-denoising runs end-to-end, in parallel, with infill + self-correct.
- **Open problem** — at the aggressive **K=4** compression, generation is locally plausible but globally incoherent (the known hard frontier of continuous-latent text diffusion). The spectral variant didn't close this — a frequency basis is a representational aid, not a capacity fix. Closing it needs **scale** (→ the TPU ask).

## Structure
```
src/codec.py        token <-> latent VAE codec                  (Stage 1)
src/diffusion.py    latent-diffusion denoiser + gen/infill/sc   (Stage 2)
src/spectral.py     frequency-domain (DCT) variant + ablation   (Stage 3)
notebooks/          Colab (TPU) driver notebooks
REPORT.md           findings + roadmap
```

## Run (Colab, TPU runtime)
1. Push this repo to GitHub (public, or you'll need a token in Colab to clone).
2. Open `notebooks/01_codec.ipynb` in Colab; Runtime → Change runtime type → **TPU**.
3. Run top to bottom.

The notebook **clones the repo fresh each session** (so it always matches your latest `git push`) and persists **data + checkpoints** to a separate `sld-poc-data/` folder on Google Drive. Checkpoints are written every ~3 min, and re-running the train cell auto-resumes after a disconnect.

**Dev loop:** edit `src/*.py` locally → `git push` → re-run the clone cell in Colab → run.

## Status
- [x] Stage 1 — codec (token ↔ latent, ~99.96% reconstruction)
- [x] Stage 2 — latent diffusion (generation, infilling, self-correction)
- [x] Spectral variant (frequency-domain diffusion) + baseline ablation
- [ ] **Next:** scale the denoiser; AST-aware & information-adaptive chunking (see [`REPORT.md`](REPORT.md))
