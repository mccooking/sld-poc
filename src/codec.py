"""
sld-poc — Stage A: the chunk codec, in JAX/Flax for TPU (v5e).

  encode:  K token IDs  ->  one latent vector z
  decode:  z            ->  K token IDs

VAE (KL) + noise injection so the latent space is smooth and the decoder
tolerates the messy vectors the diffuser will emit. Exports the latent
mean/std ("scale factor") for the diffusion stage.

Colab (TPU runtime):
  !pip -q install flax optax datasets tiktoken
  from google.colab import drive; drive.mount('/content/drive')
  !python codec.py --prep  --workdir /content/drive/MyDrive/sld-poc
  !python codec.py --train --workdir /content/drive/MyDrive/sld-poc   # re-run to resume
  !python codec.py --demo  --workdir /content/drive/MyDrive/sld-poc
"""

import argparse, os, time, pickle, functools
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# ----------------------------- config -----------------------------
K          = 4
D_EMB      = 128
D_LATENT   = 64
D_HIDDEN   = 512
BETA_KL    = 1e-3
NOISE_STD  = 0.3
BATCH      = 4096
LR         = 3e-4
STEPS      = 20000
EVAL_EVERY = 1000
CKPT_SEC   = 180
MAX_TOKENS = 50_000_000
VAL_FRAC   = 0.01
SEED       = 0


def paths(workdir):
    if workdir.startswith("/content/drive") and not os.path.ismount("/content/drive"):
        try:
            from google.colab import drive; drive.mount("/content/drive")
        except Exception as e:
            print(f"(could not auto-mount Drive: {e})")
    os.makedirs(os.path.join(workdir, "data"), exist_ok=True)
    return {
        "cache": os.path.join(workdir, "data", f"tinystories_K{K}.npy"),
        "ckpt":  os.path.join(workdir, "codec_ckpt.pkl"),   # full state, for resume
        "final": os.path.join(workdir, "codec.pkl"),        # clean params, for diffusion
        "stats": os.path.join(workdir, "latent_stats.npz"),
    }


def save_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# ----------------------------- data prep -----------------------------
def prep(P):
    if os.path.exists(P["cache"]):
        print(f"cache exists -> {P['cache']}"); return
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token
    toks = []
    for i, ex in enumerate(load_dataset("roneneldan/TinyStories", split="train", streaming=True)):
        toks.extend(enc.encode_ordinary(ex["text"])); toks.append(eot)
        if len(toks) >= MAX_TOKENS: break
        if i % 50000 == 0: print(f"  {i} stories, {len(toks):,} tokens")
    arr = np.asarray(toks[:MAX_TOKENS], dtype=np.int32)
    n = len(arr) // K
    np.save(P["cache"], arr[: n * K].reshape(n, K))
    print(f"saved {n:,} chunks -> {P['cache']}")


# ----------------------------- model -----------------------------
class CodecVAE(nn.Module):
    vocab: int

    @nn.compact
    def __call__(self, tokens, noise_std=0.0, sample=True):
        emb = nn.Embed(self.vocab, D_EMB)
        e = emb(tokens).reshape(tokens.shape[0], -1)          # [B, K*D_EMB]
        h = nn.gelu(nn.Dense(D_HIDDEN)(e))
        h = nn.gelu(nn.Dense(D_HIDDEN)(h))
        mu = nn.Dense(D_LATENT)(h)
        logvar = nn.Dense(D_LATENT)(h)
        if sample:
            eps = jax.random.normal(self.make_rng("noise"), mu.shape)
            z = mu + jnp.exp(0.5 * logvar) * eps
        else:
            z = mu
        if noise_std > 0:
            z = z + noise_std * jax.random.normal(self.make_rng("noise"), z.shape)
        d = nn.gelu(nn.Dense(D_HIDDEN)(z))
        d = nn.gelu(nn.Dense(D_HIDDEN)(d))
        d = nn.Dense(K * D_EMB)(d).reshape(tokens.shape[0], K, D_EMB)
        logits = d @ emb.embedding.T                          # tied embeddings
        return logits, mu, logvar


def make_model(vocab):
    return CodecVAE(vocab=vocab)


def decode_latents(params, z):
    """Decode latent vectors z [N, D_LATENT] -> token logits [N, K, vocab].

    Standalone so the diffusion/spectral stages can decode generated latents
    without re-running the encoder. Mirrors CodecVAE's decoder: the compact
    layers Dense_4/5/6 followed by the tied embedding (Embed_0). If you change
    the encoder/decoder layer order in CodecVAE.__call__, update these names.
    """
    p = params
    d = jax.nn.gelu(z @ p["Dense_4"]["kernel"] + p["Dense_4"]["bias"])
    d = jax.nn.gelu(d @ p["Dense_5"]["kernel"] + p["Dense_5"]["bias"])
    d = (d @ p["Dense_6"]["kernel"] + p["Dense_6"]["bias"]).reshape(z.shape[0], K, D_EMB)
    return d @ p["Embed_0"]["embedding"].T          # [N, K, vocab]


def encode_mu(params, tokens):
    """Encode token chunks [N, K] -> latent means mu [N, D_LATENT] (encoder only).

    Mirrors CodecVAE's encoder (Embed_0 + Dense_0/1 + to_mu = Dense_2). Used to build
    the latent dataset WITHOUT materializing the huge [N, K, vocab] decode logits.
    """
    p = params
    e = p["Embed_0"]["embedding"][tokens].reshape(tokens.shape[0], -1)
    h = jax.nn.gelu(e @ p["Dense_0"]["kernel"] + p["Dense_0"]["bias"])
    h = jax.nn.gelu(h @ p["Dense_1"]["kernel"] + p["Dense_1"]["bias"])
    return h @ p["Dense_2"]["kernel"] + p["Dense_2"]["bias"]      # [N, D_LATENT]


@functools.partial(jax.jit, static_argnums=0)
def train_step(model, state, batch, rng):
    def loss_fn(p):
        logits, mu, logvar = model.apply({"params": p}, batch, noise_std=NOISE_STD,
                                         sample=True, rngs={"noise": rng})
        recon = optax.softmax_cross_entropy_with_integer_labels(logits, batch).mean()
        kl = -0.5 * jnp.mean(1 + logvar - mu ** 2 - jnp.exp(logvar))
        return recon + BETA_KL * kl, (recon, kl)
    (loss, (recon, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), loss, recon, kl


@functools.partial(jax.jit, static_argnums=0)
def eval_step(model, params, batch):
    logits, _, _ = model.apply({"params": params}, batch, noise_std=0.0, sample=False)
    return (logits.argmax(-1) == batch).mean()


@functools.partial(jax.jit, static_argnums=0)
def encode_mu(model, params, batch):
    _, mu, _ = model.apply({"params": params}, batch, noise_std=0.0, sample=False)
    return mu


# ----------------------------- train -----------------------------
def train(P):
    chunks = np.load(P["cache"])
    n_val = int(chunks.shape[0] * VAL_FRAC)
    val, tr = chunks[:n_val], chunks[n_val:]
    vocab = int(chunks.max()) + 1
    print(f"chunks: {tr.shape[0]:,} train / {val.shape[0]:,} val | vocab {vocab} | {jax.devices()}")

    model = make_model(vocab)
    key = jax.random.PRNGKey(SEED)
    k_init, k_noise, key = jax.random.split(key, 3)
    params = model.init({"params": k_init, "noise": k_noise},
                        jnp.zeros((2, K), jnp.int32))["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adamw(LR))

    start = 0
    if os.path.exists(P["ckpt"]):
        ck = pickle.load(open(P["ckpt"], "rb"))
        state = state.replace(params=ck["params"], opt_state=ck["opt_state"], step=ck["step"])
        start = ck["step"]
        print(f"== resumed from step {start} ==")
    else:
        print(f"params: {n_params/1e6:.1f}M | fresh start")

    def snap(step):
        save_atomic(P["ckpt"], {"params": jax.device_get(state.params),
                                "opt_state": jax.device_get(state.opt_state), "step": step})

    last = time.time()
    for step in range(start + 1, STEPS + 1):
        idx = np.random.randint(0, tr.shape[0], BATCH)
        batch = jnp.asarray(tr[idx])
        key, sk = jax.random.split(key)
        state, loss, recon, kl = train_step(model, state, batch, sk)

        if time.time() - last > CKPT_SEC:
            snap(step); last = time.time(); print(f"  [checkpoint @ step {step} -> Drive]")
        if step % EVAL_EVERY == 0 or step == 1:
            vb = jnp.asarray(val[: min(val.shape[0], 100000)])
            acc = float(eval_step(model, state.params, vb))
            print(f"step {step:>6} | loss {float(loss):.3f} recon {float(recon):.3f} "
                  f"kl {float(kl):.2f} | tok_acc {acc*100:.3f}%")
            snap(step); last = time.time()

    save_atomic(P["final"], {"params": jax.device_get(state.params), "vocab": vocab,
                             "cfg": dict(K=K, D_EMB=D_EMB, D_LATENT=D_LATENT, D_HIDDEN=D_HIDDEN)})
    export_stats(model, state.params, tr, P)
    print(f"done. final params -> {P['final']}")


def export_stats(model, params, data, P):
    mus = []
    for i in range(0, min(data.shape[0], 500000), BATCH):
        mus.append(np.asarray(encode_mu(model, params, jnp.asarray(data[i:i + BATCH]))))
    z = np.concatenate(mus)
    np.savez(P["stats"], mean=z.mean(0), std=z.std(0) + 1e-6)
    print(f"latent stats -> {P['stats']} | std~{z.std(0).mean():.3f}")


# ----------------------------- demo -----------------------------
def demo(P, text="once upon a time there was a small cat who liked to sit on the warm mat"):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    ck = pickle.load(open(P["final"], "rb"))
    model = make_model(ck["vocab"])
    ids = enc.encode_ordinary(text); ids = ids[: (len(ids) // K) * K]
    batch = jnp.asarray(np.array(ids, np.int32).reshape(-1, K))
    logits, _, _ = model.apply({"params": ck["params"]}, batch, noise_std=0.0, sample=False)
    out = np.asarray(logits.argmax(-1)).reshape(-1).tolist()
    print("IN :", enc.decode(ids))
    print("OUT:", enc.decode(out))
    print("token match:", sum(a == b for a, b in zip(ids, out)), "/", len(ids))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    P = paths(args.workdir); print(f"workdir: {args.workdir}")
    if args.prep: prep(P)
    if args.train:
        if not os.path.exists(P["cache"]): prep(P)
        train(P)
    if args.demo: demo(P)
    if not (args.prep or args.train or args.demo): ap.print_help()
