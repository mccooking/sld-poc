"""
sld-poc — Stage 2: latent diffusion over the codec's vectors (JAX / TPU).

Trains a self-conditioned Gaussian-diffusion denoiser over grids of M codec
latents (= M*K tokens of text), then generates / infills / self-corrects.

Phases (run in-kernel from the notebook; the TPU is held by the kernel):
  build_latents(P)   encode TinyStories chunks -> standardized latent grids (local, rebuildable)
  train_diff(P)      train the denoiser (auto-resumes from Drive)
  gen(P)             unconditional generation + speed
  infill(P)          lock start & end, generate the middle      (AR can't)
  selfcorrect(P)     corrupt one latent, repair from context    (AR can't)

Needs Stage A artifacts in --workdir:  codec.pkl, latent_stats.npz, data/tinystories_K4.npy
"""

import os, time, pickle, math
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

from codec import CodecVAE, decode_latents, K, D_LATENT   # reuse the frozen codec

# ----------------------------- config -----------------------------
M            = 16          # latent positions per sequence (M*K tokens)
DIM          = 512         # denoiser width
DEPTH        = 8
HEADS        = 8
T            = 1000        # training noise levels
SAMPLE_STEPS = 50          # DDIM steps at generation time
DIFF_STEPS   = 80000
BATCH        = 256
LR           = 3e-4
EVAL_EVERY   = 2000
CKPT_SEC     = 180
N_BLOCKS_MAX = 400_000
SELFCOND_P   = 0.5
SEED         = 0


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
        "stats":   os.path.join(workdir, "latent_stats.npz"),
        "chunks":  os.path.join(workdir, "data", f"tinystories_K{K}.npy"),
        "latents": os.path.join(latdir, "sld_latents.npy"),
        "diffck":  os.path.join(workdir, "diff_ckpt.pkl"),
        "diff":    os.path.join(workdir, "diff.pkl"),
    }


def save_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: pickle.dump(obj, f)
    os.replace(tmp, path)


def load_codec(P):
    ck = pickle.load(open(P["codec"], "rb"))
    st = np.load(P["stats"])
    return ck["params"], ck["vocab"], jnp.asarray(st["mean"]), jnp.asarray(st["std"])


# ----------------------------- noise schedule -----------------------------
def cosine_abar(T):
    s = jnp.arange(T + 1) / T
    f = jnp.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    return f / f[0]                                   # abar[0]=1 (clean) ... abar[T]~0 (noise)

ABAR = cosine_abar(T)


def timestep_emb(t, dim):                             # t [B] in [0,1]
    half = dim // 2
    freqs = jnp.exp(-math.log(10000) * jnp.arange(half) / half)
    a = t[:, None] * 1000.0 * freqs[None]
    return jnp.concatenate([jnp.cos(a), jnp.sin(a)], -1)


# ----------------------------- build latent grids -----------------------------
def build_latents(P):
    if os.path.exists(P["latents"]):
        print(f"latents exist -> {P['latents']}"); return
    cparams, vocab, mean, std = load_codec(P)
    codec = CodecVAE(vocab)
    chunks = np.load(P["chunks"])                     # [N, K]
    nb = min(chunks.shape[0] // M, N_BLOCKS_MAX)
    chunks = chunks[: nb * M].reshape(nb, M, K)
    out = []
    for i in range(0, nb, 4096):
        b = jnp.asarray(chunks[i:i + 4096])           # [bb, M, K]
        bb = b.shape[0]
        mu = codec.apply({"params": cparams}, b.reshape(-1, K), noise_std=0.0, sample=False)[1]
        z = (mu - mean) / std
        out.append(np.asarray(z.reshape(bb, M, D_LATENT), dtype=np.float16))
    lat = np.concatenate(out)
    np.save(P["latents"], lat)
    print(f"built latents {lat.shape} (standardized, fp16) -> {P['latents']}")


# ----------------------------- denoiser -----------------------------
class Denoiser(nn.Module):
    @nn.compact
    def __call__(self, x_t, t, x_self):               # x_t,x_self [B,M,DL]; t [B] in [0,1]
        h = nn.Dense(DIM)(jnp.concatenate([x_t, x_self], -1))
        h = h + self.param("pos", nn.initializers.normal(0.02), (1, M, DIM))
        temb = nn.Dense(DIM)(nn.gelu(nn.Dense(DIM)(timestep_emb(t, DIM))))
        h = h + temb[:, None, :]
        for _ in range(DEPTH):
            y = nn.LayerNorm()(h)
            y = nn.MultiHeadDotProductAttention(num_heads=HEADS)(y)   # bidirectional self-attn
            h = h + y
            y = nn.LayerNorm()(h)
            y = nn.Dense(DIM)(nn.gelu(nn.Dense(DIM * 4)(y)))
            h = h + y
        return nn.Dense(D_LATENT)(nn.LayerNorm()(h))


def make_train_step(model):
    @jax.jit
    def step(state, x0, t, key):
        ab = ABAR[t][:, None, None]
        k_noise, k_sc = jax.random.split(key)
        x_t = jnp.sqrt(ab) * x0 + jnp.sqrt(1 - ab) * jax.random.normal(k_noise, x0.shape)
        tf = t.astype(jnp.float32) / T

        def loss_fn(p):
            use_sc = jax.random.bernoulli(k_sc, SELFCOND_P)
            x_self = jax.lax.cond(
                use_sc,
                lambda: jax.lax.stop_gradient(model.apply({"params": p}, x_t, tf, jnp.zeros_like(x0))),
                lambda: jnp.zeros_like(x0))
            pred = model.apply({"params": p}, x_t, tf, x_self)
            return jnp.mean((pred - x0) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss
    return step


def train_diff(P):
    if not os.path.exists(P["latents"]):
        build_latents(P)
    lat = np.load(P["latents"])                       # [Nb, M, DL] fp16
    model = Denoiser()
    key = jax.random.PRNGKey(SEED)
    k_init, key = jax.random.split(key)
    params = model.init(k_init, jnp.zeros((2, M, D_LATENT)), jnp.zeros((2,)), jnp.zeros((2, M, D_LATENT)))["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    tx = optax.adamw(optax.cosine_decay_schedule(LR, DIFF_STEPS))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    step_fn = make_train_step(model)

    start = 0
    if os.path.exists(P["diffck"]):
        ck = pickle.load(open(P["diffck"], "rb"))
        state = state.replace(params=ck["params"], opt_state=ck["opt_state"], step=ck["step"])
        start = ck["step"]
        print(f"== resumed diffuser from step {start} ==")
    else:
        print(f"diffuser: {lat.shape[0]:,} grids, M={M} | {n_params/1e6:.1f}M params | {jax.devices()} | fresh")

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
            print(f"diff step {step:>6} | mse {float(loss):.4f}")
            snap(step); last = time.time()

    save_atomic(P["diff"], {"params": jax.device_get(state.params)})
    print(f"done. diffuser -> {P['diff']}")


# ----------------------------- sampling -----------------------------
def _load_denoiser(P):
    dc = pickle.load(open(P["diff"], "rb"))
    model = Denoiser()
    apply = jax.jit(lambda p, x, tf, xs: model.apply({"params": p}, x, tf, xs))
    return dc["params"], apply


def ddim(apply, params, key, x0_known=None, known_mask=None, n=8, steps=SAMPLE_STEPS):
    """DDIM sampler. If x0_known+known_mask given, does RePaint-style inpainting."""
    shape = x0_known.shape if x0_known is not None else (n, M, D_LATENT)
    x = jax.random.normal(key, shape)
    x_self = jnp.zeros_like(x)
    km = None if known_mask is None else known_mask[None, :, None]
    ts = np.linspace(T, 1, steps).astype(int)
    for i in range(steps):
        t, t_prev = int(ts[i]), (int(ts[i + 1]) if i + 1 < steps else 0)
        if km is not None:
            key, kk = jax.random.split(key)
            noised = jnp.sqrt(ABAR[t]) * x0_known + jnp.sqrt(1 - ABAR[t]) * jax.random.normal(kk, x0_known.shape)
            x = jnp.where(km, noised, x)
        x0 = apply(params, x, jnp.full((shape[0],), t / T), x_self)
        x_self = x0
        a_t, a_prev = ABAR[t], ABAR[t_prev]
        eps = (x - jnp.sqrt(a_t) * x0) / jnp.sqrt(1 - a_t)
        x = jnp.sqrt(a_prev) * x0 + jnp.sqrt(1 - a_prev) * eps
    return x if km is None else jnp.where(km, x0_known, x)


def _to_text(cparams, enc, grids, mean, std):
    z = grids * std + mean
    logits = decode_latents(cparams, z.reshape(-1, D_LATENT))
    ids = np.asarray(logits.argmax(-1)).reshape(grids.shape[0], M * K)
    return [enc.decode([int(i) for i in row]) for row in ids]


# ----------------------------- demos -----------------------------
def gen(P, n=8):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    cparams, _, mean, std = load_codec(P)
    params, apply = _load_denoiser(P)
    key = jax.random.PRNGKey(1)
    apply(params, jnp.zeros((n, M, D_LATENT)), jnp.zeros((n,)), jnp.zeros((n, M, D_LATENT))).block_until_ready()  # warm jit
    t0 = time.time()
    grids = ddim(apply, params, key, n=n)
    grids.block_until_ready()
    dt = time.time() - t0
    print("\n=== unconditional samples ===")
    for s in _to_text(cparams, enc, grids, mean, std):
        print("  -", s.replace("\n", " ").strip()[:300])
    toks = n * M * K
    print(f"\nspeed: {toks} tokens in {dt:.2f}s = {toks/dt:.0f} tok/s")
    print(f"       CLGD used {SAMPLE_STEPS} passes for {M*K} tokens/seq; token-AR would need {M*K} sequential.")


def infill(P, n=6):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    cparams, _, mean, std = load_codec(P)
    params, apply = _load_denoiser(P)
    lat = jnp.asarray(np.load(P["latents"])[:n], dtype=jnp.float32)
    known = np.zeros(M, bool); known[:4] = True; known[-4:] = True       # keep ends, fill middle
    filled = ddim(apply, params, jax.random.PRNGKey(2), lat, jnp.asarray(known))
    print("\n=== infilling (ends fixed, middle generated) ===")
    for o, x in zip(_to_text(cparams, enc, lat, mean, std), _to_text(cparams, enc, filled, mean, std)):
        print("  ORIGINAL:", o.replace("\n", " ").strip()[:300])
        print("  INFILLED:", x.replace("\n", " ").strip()[:300]); print()


def selfcorrect(P, n=6):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    cparams, _, mean, std = load_codec(P)
    params, apply = _load_denoiser(P)
    lat = jnp.asarray(np.load(P["latents"])[:n], dtype=jnp.float32)
    j = M // 2
    corrupted = lat.at[:, j].set(jax.random.normal(jax.random.PRNGKey(3), lat[:, j].shape))
    known = np.ones(M, bool); known[j] = False                           # repair slot j from context
    repaired = ddim(apply, params, jax.random.PRNGKey(4), lat, jnp.asarray(known))
    print(f"\n=== self-correction (slot {j} corrupted, then repaired) ===")
    for o, c, r in zip(_to_text(cparams, enc, lat, mean, std),
                       _to_text(cparams, enc, corrupted, mean, std),
                       _to_text(cparams, enc, repaired, mean, std)):
        print("  ORIGINAL :", o.replace("\n", " ").strip()[:300])
        print("  CORRUPTED:", c.replace("\n", " ").strip()[:300])
        print("  REPAIRED :", r.replace("\n", " ").strip()[:300]); print()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for a in ("latents", "diff", "gen", "infill", "selfcorrect"):
        ap.add_argument(f"--{a}", action="store_true")
    args = ap.parse_args()
    P = paths(args.workdir); print(f"workdir: {args.workdir}")
    if args.latents: build_latents(P)
    if args.diff: train_diff(P)
    if args.gen: gen(P)
    if args.infill: infill(P)
    if args.selfcorrect: selfcorrect(P)
