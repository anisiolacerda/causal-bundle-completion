# DDBC Reproduction Spec (no-placeholder)

> **Authority note.** This spec reproduces the **runnable artifact** — the public repo
> `github.com/LiAi16/DDBC` as cloned and read (recon 2026-06-24) — **not** the paper's prose.
> Where the paper and the code disagree, **the code governs** (the paper's L=4 trained-RVQ /
> C=128 / linear-schedule / GD+STE description is superseded; see the MISMATCH section in
> `literature_review/ddbc.md`). All citations are `file:line` relative to the cloned repo root.
> Paper-equation references are to `literature_review/ddbc.md`.

This is the spec to re-implement to land on the DDBC headline (Spotify-30 F1 ≈ 0.282; the
control track target is Spotify-90 F1 ≈ 0.287). Build to *this*, then add the PIC-Diff theme at
the marked conditioning slot.

---

## 0. One-paragraph mental model

DDBC = a fork of **MDLM/SUBS continuous-time masked discrete diffusion** (Sahoo et al. 2024) over
**offline FAISS-RVQ item code tokens**. The tokenizer is **not** the paper's trainable VQ-VAE; it
is a frozen FAISS `IndexResidualQuantizer` (3 semantic levels) plus a deterministic 4th
"dedup" counter digit, dumped to disk once and reloaded. The denoiser is a tiny bidirectional
DDiT (6 blocks / hidden 64 / 8 heads), trained continuous-time (`T=0`) with `time_conditioning=False`
(so it never actually sees the noise level), under a loglinear schedule, with weight-EMA 0.9999.
Inference is semi-AR ancestral DDPM-cache sampling (`sampling.steps=25`) with a hard per-level
token-validity mask and a clamp that freezes observed/revealed positions.

---

## 1. RVQ tokenizer

**Type: offline FAISS residual quantizer, frozen. NOT trained with the diffusion model.**

| Item | Value | Evidence |
|---|---|---|
| Quantizer | `faiss.IndexResidualQuantizer(d, rq_n_codebooks, n_bits, faiss.METRIC_INNER_PRODUCT)` | `tokenizer.py:198-205` |
| Metric | **inner product** (NOT L2; paper Eq.1 says L2) | `tokenizer.py:204` |
| Semantic levels | `rq_n_codebooks = 3` | `configs/config.yaml:127` |
| Codebook size per level | `rq_codebook_size = 256` (**NOT 128**) | `configs/config.yaml:128` |
| `n_bits` | `int(np.log2(256)) = 8` | `tokenizer.py:196` |
| Codebook init / update | FAISS internal training (`index.train(sent_embs)`); **no GD, no STE, no commitment loss, no EMA-on-codebook, no k-means-from-scratch in the active RVQ path** | `tokenizer.py:209, 200-235` |
| Codebook extraction | `faiss.vector_to_array(rq.codebooks).reshape(M, ks, d)` → merged `[M·ks, d]`, saved to `clhe_weight.npy` | `tokenizer.py:226-235` |
| Coupling to diffusion | **precomputed offline + frozen**, loaded from `clhe_sid.npy` / `clhe_token.json` / `clhe_weight.npy` if present; diffusion only ever consumes integer token sequences | `tokenizer.py:273-313, 285-311` |
| Encoder feature | frozen **CLHE**, `feature_type=clhe`, `sent_emb_pca=-1` (no PCA; raw `clhe.pt` loaded) | `configs/config.yaml:115`; `tokenizer.py:294-296` |
| Embedding dim `d` | `sent_embs.shape[-1]` = **768** (CLHE; baseline configs set `sent_emb_dim:768`) | `tokenizer.py:201`; `baseline/genrec/models/TIGER/config.yaml:9` |
| Codebook → token embedding | merged codebook copied into the DiT input `EmbeddingLayer` (`embedding.data[1:weight.shape[0]+1]`) | `models/dit.py:298-302` |

**Re-impl note (paper vs code).** If you instead implement the *paper's* trainable RVQ (GD + STE +
commitment, Eq.2, L2 metric), you are building a different tokenizer than the one that produced the
headline numbers. To reproduce the artifact, use FAISS-RQ as above. The paper's commitment weight β
is **N/A** here (no commitment term exists; config `beta1/beta2` are AdamW moments, `config.yaml:62-63`).

### 1a. Dedup (4th digit) + CODE2ITEM

- **Dedup digit** = 1-based running occurrence count of the 3-tuple semantic key, in catalog-id
  order; deterministic; overflow (`max_conflict > rq_codebook_size`) raises. `tokenizer.py:173-187`:
  - `item2sem_ids[i] = (*sem_ids[i].tolist(), len(sem_id2item[str_id]))` (`tokenizer.py:179`).
- So each item's code tuple has **width 4 = 3 FAISS semantic + 1 dedup counter** (the 4th is NOT a
  quantized residual level).
- **Reconstruction uses only the 3 semantic levels** (sum of codebook rows): `tokenizer.py:240-255`.
- **CODE2ITEM (eval):** exact inverse dict `token2id = {tuple(v): k}` (`evaluator.py:33-34`), with a
  cosine-NN fallback over a candidate pool: predicted 3-code tuple → CLHE feature (exact via
  `token2id`, else reconstructed) → cosine-NN against a ρ=100 shortlist. `evaluator.py:220-262`.

### 1b. Per-level vocab banding (token id offsets)

`tokenizer.py:261-267` — each level occupies a disjoint integer band (0 reserved for pad/bos):
- level d (d=0,1,2): `tokens[d] += rq_codebook_size * d + 1`
- dedup digit: `tokens[-1] += rq_codebook_size * n_digit` (n_digit=3)

With **C=256**: level1 ∈ [1,256], level2 ∈ [257,512], level3 ∈ [513,768], dedup ∈ [769,1024].
The inference legal-mask and evaluator **hard-code this 256-banded layout** (`diffusion.py:786-801`;
`evaluator.py:213-217`) — the code is wired to C=256, not C=128.

---

## 2. Serialization

**Flag tokens** (`tokenizer.py:78-83`, for C=256):
- `bos = 0`
- `boi = 4*rq_codebook_size + 1 = 1025` (note the literal `4*`, leaving room for the dedup band)
- `eos = boi + 1 = 1026`
- `vocab_size = 1027` (`tokenizer.py:586`); the diffusion model appends a fresh `mask_index = vocab_size`
  (`diffusion.py:103-104`).

**Per-bundle sequence** (`_tokenize_once`, `tokenizer.py:339-354`):
```
input_ids = [bos]
for i in item_seq:
    input_ids.extend([boi] + i_token)   # i_token = 4 digits
input_ids.append(eos)
```
- Each item = a **5-token block** `[boi, d1, d2, d3, d4]`.
- **Length formula:** `U = 2 + |b|·(L+1)` with L=4 → `U = 2 + 5·|b|`.
  - spotify30 train: `model.length=152 = 2 + 5·30` (`scripts/train.sh:11`).
  - eval input half = `ceil(30/2)=15` items: `model.length=77 = 2 + 5·15` (`scripts/test.sh:11`).
- **Test ground-truth labels drop the boi** (`test_gt=True`): emitted as bare 4-digit groups, no
  bos/boi/eos. `tokenizer.py:348-349`. (The evaluator reshapes preds into 5-blocks then strips boi
  to align with the 4-block labels — `evaluator.py:269-274`.)

---

## 3. Diffusion (training)

| Item | Value | Evidence |
|---|---|---|
| Parameterization | `subs` (MDLM SUBS) | `configs/config.yaml:13`; `_subs_parameterization` `diffusion.py:279-296` |
| Time | **`T=0` continuous-time** (forward draws continuous `t∈[eps,1]`; discretization block skipped) | `configs/config.yaml:15`; `diffusion.py:1029-1037, 1077-1086` |
| time_conditioning | **False** — σ zeroed before backbone (`sigma = torch.zeros_like(sigma)`), so the denoiser is **not** conditioned on t/σ at all | `configs/config.yaml:14`; `diffusion.py:328-329` |
| Noise schedule | **loglinear**, `LogLinearNoise(eps=1e-3)` (hard-coded eps; `loglinear.yaml` sigma_min/max are **unused** — constructed with no args) | `configs/config.yaml:7`; `noise_schedule.py:18, 126-144` |
| Forward corruption | i.i.d. absorbing mask: `move_indices = rand < move_chance; xt = where(move_indices, mask_index, x)` | `diffusion.py:681-692` |
| Flags never masked | `move_chance.masked_fill(isin(x0, [eos,bos,boi]), 0.0)` | `diffusion.py:1098-1100` |

**Noise formulas** (`noise_schedule.py:140-144`):
- total noise: `σ(t) = −log(1 − (1−ε)·t)`, ε=1e-3
- rate: `dσ/dt = (1−ε)/(1−(1−ε)·t)`
- move-chance (mask prob): `move_chance = 1 − e^{−σ} = (1−ε)·t` → keep-prob `α_t = 1 − (1−ε)·t`
  (essentially linear in continuous t; this is what makes the loglinear marginal ≈ the paper's linear one).

**Exact loss — continuous-time SUBS NELBO** (the `T=0` branch; `diffusion.py:1124-1162`):
```
log_p_theta = gather(model_output, dim=-1, index=x0)        # log-prob of the TRUE clean token
loss = - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
nlls = loss                                                 # attention_mask is COMMENTED OUT
count = loss.shape[1]                                       # = full padded seq length (NOT masked count)
token_nll = nlls.sum() / count
```
- SUBS forces `log_p_theta = 0` at every **unmasked** position (`diffusion.py:295`), so only masked
  code positions carry gradient; flags contribute 0 (never masked). This is masked-token weighted CE
  of the clean token, in **continuous-time** form (weight `dσ/expm1(σ)`), NOT the discrete `1/T·ΣCE`
  of the paper's Eq. 7.
- `_reconstruction_loss` and `_d3pm_loss` exist but are **not** on the `T=0` path
  (`diffusion.py:1066-1075, 350-379, 1109, 1116`).

---

## 4. Inference sampler

**Entry point used for construction eval = semi-AR** (`main.py:143-153`), NOT `_sample`:
```
stride_length = input_ids.shape[1]   # input length == output length
num_strides   = 1
model.restore_model_and_semi_ar_sample(input_ids, stride_length, num_strides, dt=1/config.sampling.steps)
```
- **Steps:** `sampling.steps = 25` (set by `scripts/train.sh:13` / `scripts/test.sh`); `dt = 1/25`,
  `num_steps = int(1/dt) = 25` reverse iterations over continuous t descending 1→0
  (`main.py:153`; `diffusion.py:1212, 1228`). *(Base `config.yaml` default is 256 — overridden to 25
  by the run scripts; reproduce with 25.)*
- The tail block `b_y` = `stride_length` mask tokens appended to the clamped observed prefix
  (`x = cat(target, x_tmp)`, `diffusion.py:1221-1224`).

**Per-step update — ancestral DDPM-cache** (`_ddpm_caching_update`, `diffusion.py:815-838`):
```
p_x0 = self.forward(x, sigma_t).exp()
q_xs = p_x0 * (move_chance_t - move_chance_s)
q_xs[:, :, mask_index] = move_chance_s[:, :, 0]
q_xs = self._apply_illegal_mask(q_xs, tokenizer, stride_length)   # per-level validity (see 4c)
_x   = _sample_categorical(q_xs)                                  # Gumbel-argmax
copy_flag = (x != mask_index)
return p_x0, copy_flag * x + (1 - copy_flag) * _x                 # CLAMP (see 4a)
```

### 4a. Clamp (observed + revealed frozen)
`copy_flag` copies every **non-mask** position verbatim (all of `b_x`, all flags, all already-revealed
`b_y` tokens). Observed/flag tokens are never the mask id → never overwritten; once a `b_y` position is
revealed it is frozen (**no re-masking**). `diffusion.py:837-838`. **Matches** paper clamp / footnote-2.

### 4b. Reveal rule — **stochastic, NOT top-k/entropy/ratio**
There is **no** explicit reveal-set selection. Each masked position is sampled from the categorical
`{stay-mask (prob = move_chance_s), decode (prob ∝ p_x0·(mc_t−mc_s))}`; the number revealed per step is
random, governed by `move_chance_t − move_chance_s ≈ dt·(1−ε)`. Over 25 steps mask prob → ~0, so all get
revealed. This is the inherited MDLM ancestral sampler — the paper's top-k/entropy/η options are not
what runs. `diffusion.py:815-835`.

### 4c. Per-level token-validity mask (`_apply_illegal_mask`, `diffusion.py:736-813`)
Applied **every step**, **only at positions being unmasked this step**, on block structure `k=5`
(`pos % 5`), with C=256 hard-coded bands; invalid logits set to `-1e9`:
- `pos == start | pos == seq_len-1 | pos % 5 == 3` → skip (bos / eos / boi)
- `pos % 5 == 4` → level-1 legal `[1, C+1)`
- `pos % 5 == 0` → level-2 legal `[C+1, 2C+1)`
- `pos % 5 == 1` → level-3 legal `[2C+1, 3C+1)`
- `pos % 5 == 2` → dedup level `[3C+1, 4C+1)` (left effectively free per Lane recon)
- BOS/EOS endpoints always allowed (`temp_mask[0]=temp_mask[-1]=True`).
**Matches** paper Eq. step (c). `codebook_size` read from `tokenizer.config['rq_codebook_size']`
(`diffusion.py:751`).

### 4d. Cache mechanics
With `time_conditioning=False`, `p_x0` depends only on `x`; cache is reused while `x` is unchanged and
invalidated on any reveal (`if not allclose(x_next, x): p_x0_cache = None`, `diffusion.py:1232-1234`).
EMA weights are swapped in for sampling/validation (`diffusion.py:932-943, 1262-1278`).

---

## 5. DiT architecture (`model=small`)

`configs/model/small.yaml:3-9`: `hidden_size=64`, `cond_dim=128`, `length=32` (overridden per dataset),
`n_blocks=6`, `n_heads=8`, `scale_by_sigma=True`, `dropout=0.1`, `tie_word_embeddings=False`.

Wiring (`models/dit.py:330-374`):
- `vocab_embed = EmbeddingLayer(64, vocab_size)`; `sigma_map = TimestepEmbedder(128)`;
  `rotary_emb = Rotary(64//8 = 8)` (per-head dim **8**); 6× `DDiTBlock`; `DDitFinalLayer(64, vocab, 128)`.
- `DDiTBlock` (`dit.py:215-289`): pre-LN, RoPE attention via `flash_attn_varlen_qkvpacked_func`
  with `causal=False` (**bidirectional**, `dit.py:273-274`), mlp_ratio=4 GELU-tanh, **adaLN-zero**
  modulation (6 chunks from `cond_dim`, zero-init `dit.py:234-235`).
- **Param count:** paper claims 0.79M. Not re-derived mechanically (no torch in sandbox); architecture
  (6/64/8) matches; vocab=1027 makes the embedding terms vocab-dependent. **Verify at repro with a
  one-line `sum(p.numel())`** — do not assert 0.79M without it.
- **Do NOT seed from `models/config.json` / `configuration_mdlm.py`** — those are the OWT-text MDLM
  checkpoint (hidden 768 / 12 blocks / 12 heads / vocab 50258), not the bundle model.

### 5a. PIC-Diff conditioning slot (where the iVAE theme enters)
`cond_dim=128` currently carries **only** the timestep embedding `c = silu(sigma_map(sigma))`
(`dit.py:367`), and under `time_conditioning=False` σ→0 so `c` is a constant bias — the slot is
**information-free today**. Inject the theme/user vector `u∈R^128` by adding into `c` **before** the
block loop:
```
c = F.silu(self.sigma_map(sigma)) + proj_theme(theta)   # proj_theme: Linear(theme_dim → cond_dim=128)
```
This FiLMs every `DDiTBlock.adaLN_modulation` and the `DDitFinalLayer` — global, per-sequence
conditioning that does **not** touch per-token logits (so it cannot violate the per-level validity
filter; the two are orthogonal). Because adaLN is zero-init, `proj_theme` starts inert: the **CI-gate**
(ablation delta, τ@0 / set-correlation under ablation, gradient-flow magnitude through `proj_theme`) is
exactly what proves it becomes load-bearing. Note the `dit.py` standalone `forward` does NOT apply the
σ→0 zeroing (only `DITBackbone.forward` in `modeling_mdlm.py:387-388` does) — confirm which forward
`diffusion.py` calls; it is load-bearing for whether the slot is truly inert.

---

## 6. Optimizer / schedule / training

| Item | Value | Evidence |
|---|---|---|
| Optimizer | `torch.optim.AdamW`, lr **3e-4**, betas (0.9, 0.999), eps 1e-8, **weight_decay 0** (→ effectively Adam) | `configs/config.yaml:61-64`; `diffusion.py:560-567` |
| LR scheduler | `transformers.get_constant_schedule_with_warmup`, **500 warmup steps**, constant after | `configs/lr_scheduler/constant_warmup.yaml:1-2` |
| Max steps | **20000** | `configs/config.yaml:75` |
| Global batch size | **512** (per-device `div_up(512, devices)`) | `configs/config.yaml:22` |
| Grad clip | 1.0; `val_check_interval=32` | `configs/config.yaml:72, 79` |
| Weight-EMA | **0.9999**, over `backbone + noise` params (NOT codebook), annealed `min(decay,(1+n)/(10+n))`, swapped in for val/sample | `configs/config.yaml:41`; `diffusion.py:157-161, 274-275`; `models/ema.py:40-44` |
| Precision | **bf16** (forward internally force-casts backbone to fp32, `dit.py:371`) | `configs/config.yaml:73` |
| Strategy | DDP, `find_unused_parameters=False` | resolved run cfg `004302/.hydra/config.yaml:147-149` |

---

## 7. Data / split / candidate protocol

| Item | Value | Evidence |
|---|---|---|
| Dataset | HF `xhLiu/BundleConstruction` (`spotify`); `tokenizer_name_or_path: MDLMTokenizer` | `configs/data/spotify.yaml` |
| seq_len (k) | **30** for spotify30; bundles shorter than k are **dropped** | `configs/config.yaml:117`; `tokenizer.py:560-562` |
| Truncation (valid/test) | **deterministic prefix `[:seq_len]`** (NOT the paper's random-start window) | `tokenizer.py:366, 396, 466` |
| Augmentation (train only) | `swap_ratio=0.4` (Spotify): `int(len·0.4)` adjacent swaps, each → a fresh copy then a **random-start** `seq_len` window | `configs/config.yaml:119`; `dataset.py:69-91` |
| Observed/label split | `index = ceil(len/2)`; input = `item_seq[:index]`, label = `item_seq[index:]` (test_gt) → **prefix-observed, 1:1** | `tokenizer.py:369-371` |
| `cir` | **=1** = plain per-item RVQ (do not cluster items→components). **NOT** the input:predict ratio (that is the fixed `ceil(len/2)` split). | `configs/config.yaml:118`; `main.py:305-311` |
| Candidate pool ρ | **100** = `|b_y|` truth + `99·|b_y|` random (re-drawn per attempt) | `evaluator.py:237` |
| Retrieval | cosine-NN of CLHE features over the ρ=100 shortlist, top-1/slot, sampling-without-replacement | `evaluator.py:248-262` |

### 7a. Metrics — port carefully
- **recall/precision/jaccard** operate on sets of item-id tuples — **match** paper. `evaluator.py:86-132`.
- **OAS:** repo computes a **`1−cos` Hungarian distance** (lower=better) normalized by **`len(pred)`**
  (`evaluator.py:134-152, 181-200`) — the paper's OAS is a **cosine similarity** (higher=better)
  normalized by **target size**. **MISMATCH in sign/normalization**: `OAS_paper ≈ 1 − oas_mean` only if
  `|pred|=|b_y|` and dummy padding matches. Reconcile before reporting.
- **F1 is BUGGED** in the repo (`2·R·R/(R+R) = R`, recall twice — `evaluator.py:37-38`). It is not in
  the default metric list (`config.yaml:112` = `[recall,precision,hit,hitf,jaccard,oas]`). **Recompute
  F1 = 2PR/(P+R) yourself** from P and R.
- topk default `[1]`; headline is single-attempt (`config.yaml:111`; `main.py:145-158`).

---

## 8. Reproduction checklist (artifact-faithful, not paper-faithful)

1. FAISS `IndexResidualQuantizer`, inner-product, **3 levels × C=256**, n_bits=8, frozen offline +
   appended dedup-counter digit → 4-tuple. (NOT trainable RVQ / C=128 / L2.)
2. Serialize `[bos] (boi d1 d2 d3 d4)* [eos]`, `length = 2 + 5k`; labels drop boi.
3. SUBS continuous-time (`T=0`), `time_conditioning=False`, **loglinear** noise ε=1e-3, absorbing mask,
   flags never masked, continuous-time NELBO weighted by `dσ/expm1(σ)`, normalized by full seq length.
4. Tiny DDiT (6/64/8, bidirectional, adaLN-zero); inject PIC-Diff theme into `c` (§5a).
5. AdamW lr 3e-4 (wd 0), 500-step warmup constant, 20k steps, batch 512, grad-clip 1.0, EMA 0.9999, bf16.
6. Semi-AR DDPM-cache inference, **25 steps**, stochastic reveal, clamp non-mask positions, per-level
   `-1e9` validity mask on positions unmasked this step.
7. Eval: prefix-observed 1:1, deterministic `[:k]` truncation, ρ=100 cosine-NN retrieval; **recompute
   F1**; **reconcile OAS sign/normalization**.
