"""
sld-poc — Stage 3: SPECTRAL latent diffusion (JAX / TPU).

Same denoiser as Stage 2, but it diffuses the latents in the FREQUENCY DOMAIN:
an orthonormal DCT is applied across the M positions, so the model first nails
low-frequency (global) structure and fills high-frequency (local) detail last.

Pipeline:
  encode tokens -> mu  ->  DCT across positions  ->  per-frequency standardize
  -> diffuse the spectral coefficients -> (un-standardize -> inverse-DCT -> decode)

The deliverable is a BASELINE-vs-SPECTRAL comparison: run Stage 2's gen and this
one, and compare coherence. (Infill/self-correct are position-space ops and live
in Stage 2; here the M slots are frequencies, so we focus on generation.)

Reuses Stage 2's Denoiser/train-step/sampler; needs Stage A's codec.pkl.
"""

import os, time, pickle
import numpy as np
import jax, jax.numpy as jnp
from flax.training import train_state
import optax

from codec import encode_latents, decode_latents, K
from diffusion import (Denoiser, make_train_step, ddim, _load_denoiser, save_atomic,
                       M, D_LATENT, T, SAMPLE_STEPS, DIFF_STEPS, BATCH, LR,
                       EVAL_EVERY, CKPT_SEC, SEED)

SPEC_BLOCKS = 200_000   # cap (DCT build holds float32 in RAM)


def paths(workdir):
    if workdir.startswith("/content/drive") and not os.path.ismount("/content/drive"):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception as e:
            print(f"(could not auto-mount Drive: {e})")
    latdir = "/content" if workdir.startswith("/content/drive") else os.path.join(workdir, "data")
    os.makedirs(latdir, exist_ok=True)
    return {
        "codec":   os.path.join(workdir, "codec.pkl"),
        "chunks":  os.path.join(workdir, "data", f"tinystories_K{K}.npy"),
        "latents": os.path.join(latdir, "sld_spectral_latents.npy"),
        "stats":   os.path.join(workdir, "spectral_freqstats.npz"),
        "diffck":  os.path.join(workdir, "spectral_ckpt.pkl"),
        "diff":    os.path.join(workdir, "spectral.pkl"),
    }


# ----------------------------- DCT across positions -----------------------------
def _dct_matrix(m):
    n = jnp.arange(m)
    k = jnp.arange(m)[:, None]
    Cm = jnp.cos(jnp.pi * (2 * n + 1) * k / (2 * m)) * jnp.sqrt(2.0 / m)
    Cm = Cm.at[0].set(Cm[0] / jnp.sqrt(2.0))          # orthonormal DCT-II
    return Cm

C = _dct_matrix(M)                                     # [M, M]

def dct(z):   return jnp.einsum("kn,bnd->bkd", C, z)   # positions -> frequencies
def idct(S):  return jnp.einsum("kn,bkd->bnd", C, S)   # frequencies -> positions


def _load_codec_params(P):
    ck = pickle.load(open(P["codec"], "rb"))
    return jax.tree_util.tree_map(jnp.asarray, ck["params"]), ck["vocab"]


# ----------------------------- build spectral latents -----------------------------
def build_latents(P):
    if os.path.exists(P["latents"]):
        print(f"spectral latents exist -> {P['latents']}"); return
    cparams, vocab = _load_codec_params(P)
    chunks = np.load(P["chunks"])
    nb = min(chunks.shape[0] // M, SPEC_BLOCKS)
    chunks = chunks[: nb * M].reshape(nb, M, K)
    S = np.empty((nb, M, D_LATENT), np.float32)
    for i in range(0, nb, 4096):
        b = jnp.asarray(chunks[i:i + 4096])
        mu = encode_latents(cparams, b.reshape(-1, K)).reshape(b.shape[0], M, D_LATENT)
        S[i:i + b.shape[0]] = np.asarray(dct(mu))     # DCT across positions
    fmean, fstd = S.mean(0), S.std(0) + 1e-6          # per-frequency stats [M, DL]
    Snorm = ((S - fmean) / fstd).astype(np.float16)
    np.save(P["latents"], Snorm)
    np.savez(P["stats"], fmean=fmean, fstd=fstd)
    print(f"built spectral latents {Snorm.shape} (DCT across positions, per-freq standardized) -> {P['latents']}")


# ----------------------------- train (reuses Stage 2 denoiser) -----------------------------
def train_diff(P):
    if not os.path.exists(P["latents"]):
        build_latents(P)
    lat = np.load(P["latents"])
    model = Denoiser()
    key = jax.random.PRNGKey(SEED)
    k_init, key = jax.random.split(key)
    params = model.init(k_init, jnp.zeros((2, M, D_LATENT)), jnp.zeros((2,)), jnp.zeros((2, M, D_LATENT)))["params"]
    tx = optax.adamw(optax.cosine_decay_schedule(LR, DIFF_STEPS))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    step_fn = make_train_step(model)

    start = 0
    if os.path.exists(P["diffck"]):
        ck = pickle.load(open(P["diffck"], "rb"))
        state = state.replace(params=ck["params"], opt_state=ck["opt_state"], step=ck["step"])
        start = ck["step"]
        print(f"== resumed spectral diffuser from step {start} ==")
    else:
        print(f"spectral diffuser: {lat.shape[0]:,} grids, M={M} | {jax.devices()} | fresh")

    def snap(step):
        save_atomic(P["diffck"], {"params": jax.device_get(state.params),
                                  "opt_state": jax.device_get(state.opt_state), "step": step})

    last = time.time()
    for step in range(start + 1, DIFF_STEPS + 1):
        idx = np.random.randint(0, lat.shape[0], BATCH)
        x0 = jnp.asarray(lat[idx], dtype=jnp.float32)
        t = jnp.asarray(np.random.randint(1, T + 1, BATCH))
        key, sk = jax.random.split(key)
        state, loss = step_fn(state, x0, t, sk)

        if time.time() - last > CKPT_SEC:
            snap(step); last = time.time(); print(f"  [checkpoint @ step {step} -> Drive]")
        if step % EVAL_EVERY == 0 or step == 1:
            print(f"spec step {step:>6} | mse {float(loss):.4f}")
            snap(step); last = time.time()

    save_atomic(P["diff"], {"params": jax.device_get(state.params), "DL": D_LATENT})
    print(f"done. spectral diffuser -> {P['diff']}")


# ----------------------------- generate -----------------------------
def gen(P, n=8):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    cparams, _ = _load_codec_params(P)
    st = np.load(P["stats"]); fmean = jnp.asarray(st["fmean"]); fstd = jnp.asarray(st["fstd"])
    apply = _load_denoiser(P)                          # loads spectral.pkl
    apply(jnp.zeros((n, M, D_LATENT)), jnp.zeros((n,)), jnp.zeros((n, M, D_LATENT))).block_until_ready()
    t0 = time.time()
    grids = ddim(apply, jax.random.PRNGKey(1), n=n)    # sampled spectral coeffs (standardized)
    grids.block_until_ready()
    dt = time.time() - t0
    mu = idct(grids * fstd + fmean)                    # un-standardize -> inverse-DCT -> latents
    logits = decode_latents(cparams, mu.reshape(-1, D_LATENT))
    ids = np.asarray(logits.argmax(-1)).reshape(n, M * K)
    print("\n=== SPECTRAL samples ===")
    for row in ids:
        print("  -", enc.decode([int(i) for i in row]).replace("\n", " ").strip()[:300])
    toks = n * M * K
    print(f"\nspeed: {toks} tokens in {dt:.2f}s = {toks/dt:.0f} tok/s | {SAMPLE_STEPS} passes")
    print("  (compare these against Stage 2's `gen` output — that's the baseline-vs-spectral result)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for a in ("latents", "diff", "gen"):
        ap.add_argument(f"--{a}", action="store_true")
    args = ap.parse_args()
    P = paths(args.workdir); print(f"workdir: {args.workdir}")
    if args.latents: build_latents(P)
    if args.diff: train_diff(P)
    if args.gen: gen(P)
