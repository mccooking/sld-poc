# sld-poc

A **proof-of-concept** continuous-latent **diffusion language model**, with a **spectral** (frequency-domain) variant — built in JAX/Flax for **TPU (v5e)**.

## Idea
Compress every *K* tokens into one continuous vector (a CALM-style codec), then **generate text by denoising a grid of those vectors in parallel** — diffusion, instead of one token at a time. The spectral variant diffuses the latents in the **frequency domain** (a DCT across positions), so generation proceeds coarse → fine for better global structure.

Capabilities it demonstrates that autoregressive models can't: **parallel generation, infilling, and self-correction.**

> Scope: this is a deliberately small PoC (TinyStories scale) to validate the approach and characterize the compression-vs-quality frontier. The full-scale follow-up will live in a separate repo.

## Structure
```
src/codec.py        token <-> latent VAE codec                (Stage A)   ✅
src/diffusion.py    latent-diffusion denoiser + sampler       (Stage B/C) ⏳
src/spectral.py     DCT-across-positions transform (the novel bit)         ⏳
notebooks/          Colab (TPU) driver notebooks
```

## Run (Colab, TPU runtime)
1. Push this repo to GitHub (public, or you'll need a token in Colab to clone).
2. Open `notebooks/01_codec.ipynb` in Colab; Runtime → Change runtime type → **TPU**.
3. Run top to bottom.

The notebook **clones the repo fresh each session** (so it always matches your latest `git push`) and persists **data + checkpoints** to a separate `sld-poc-data/` folder on Google Drive. Checkpoints are written every ~3 min, and re-running the train cell auto-resumes after a disconnect.

**Dev loop:** edit `src/*.py` locally → `git push` → re-run the clone cell in Colab → run.

## Status
- [x] Stage A — codec (token ↔ latent, ~99.8% reconstruction)
- [ ] Stage B/C — latent diffusion (generation, infilling, self-correction)
- [ ] Spectral variant (frequency-domain diffusion)
