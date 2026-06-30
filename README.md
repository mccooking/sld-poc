# sld-poc

A proof of concept for generating text with diffusion in a continuous latent space, plus a frequency-domain variant. Written in JAX, trained on TPU, at small (TinyStories) scale.

It works in three parts. A codec turns every 4 tokens into one vector and can rebuild them. A diffusion model then generates whole sequences of these vectors at once, starting from noise and cleaning it up over a few passes, rather than emitting tokens left to right. The spectral variant runs that same diffusion on a DCT of the vectors, so the model settles the coarse structure first and fills in detail afterwards.

![Pipeline: tokens to latent vectors, diffusion, and back to tokens](docs/pipeline.png)

Since one vector carries 4 tokens, the diffuser deals with a quarter as many positions as a token-level model, and because it denoises in parallel it isn't locked into left-to-right order. The same property lets it infill and correct its own output, which autoregressive models can't.

Findings and next steps are in [REPORT.md](REPORT.md). The short version:

- The codec reconstructs tokens at about 99.96%.
- The full generate-by-denoising loop runs end to end.
- At 4 tokens per vector the output has real words and local phrases but no overall coherence, and the spectral variant didn't change that. Getting coherent text looks like a question of model size, not of the transform.

## Layout

    src/codec.py       the codec
    src/diffusion.py   the diffusion denoiser, plus generate / infill / self-correct
    src/spectral.py    the frequency-domain variant and comparison
    notebooks/         one Colab TPU notebook per stage
    REPORT.md          findings and roadmap

## Running it

Open a notebook in Colab on a TPU runtime and run it top to bottom. Each one clones the repo, trains, and writes checkpoints to Google Drive so you can stop and pick up later. Order: `01_codec`, then `02_diffusion`, then `03_spectral`.

## Status

The codec, the diffusion model, and the spectral variant are all done. Next is a larger denoiser and two chunking ideas, structure-aware and variable-size, both described in [REPORT.md](REPORT.md).
