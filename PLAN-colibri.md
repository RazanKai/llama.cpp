# PLAN-colibri — Inkling 2 Mini long context in Colibri

Status: draft
Target: **250K-token context (MVP)**, **1M-token context (Stage 2)**
Engine: Colibri `dev` (v2.2.0), `inkling2` family in `c/inkling.c` + `c/inkling2.h`
Model: `unsloth/Inkling-2-Mini-GGUF` UD-IQ3_XXS, imported with `coli import` (`arch inkling2`)
Host: RTX 5070 12 GB, i5-10400F (6C/12T, AVX2), 23 GB RAM, NVMe

## 1. Why this plan exists

Colibri already implements the hierarchical placement we wanted from Strata:

- hot experts in VRAM (expert tier);
- warm experts in RAM (LRU + pinned history);
- cold experts streamed from NVMe;
- CPU and device compute overlapped;
- prefix reuse for agent turns.

The dev branch also already supports Inkling 2 Mini: looped layers, attention residuals,
relative-position bias, short convolutions, both GGUF layouts (`inkling2` and
`inkling2_loop`), vision, tools, and the MTP/DSpark drafter. The missing piece for long context is
the **KV cache**: the `inkling2` engine keeps `float **K, **V` (fp32) and `CTX_MAX` defaults to
8192. That is what this plan fixes.

The llama.cpp plan (`PLAN.md`) has q4 KV today but no expert cache. Colibri has the expert cache
but fp32 KV. The 64K baseline below shows llama.cpp is currently **faster**, not slower; this plan
is about making Colibri competitive at 250K/1M, not a claim that it already is.

## 2. Measured on this box

Build: both backends work.

- **CUDA** is the primary path: `make -C c inkling CUDA=1 CUDA_HOME=/opt/cuda` builds with
  CUDA 13.4 and GCC 16.2.1 (the Makefile's warning about GCC 16 is stale here; a test compile
  and the real build both succeed). The binary links `libcudart`, `libcublasLt`, `libcublas`.
- **Vulkan** is the fallback: `make -C c inkling VK=1` also works and was used before CUDA was
  tested.

Smoke test (5-token prompt, 8 generated, `CTX_MAX=2048`):

| Backend | Prefill | Decode | Notes |
|---|---:|---:|---|
| CUDA, warm history | **0.2 s** | **29.97 tok/s** | tier 2371 experts / 7.70 GiB; dense 0.94 GB on card |
| Vulkan, cold | 18.7 s | 5.90 tok/s | shader/pipeline + tier warm |
| Vulkan, warm | 0.9 s | 24.84 tok/s | tier 2540 experts / 8.24 GiB |
| CPU-only | 1.6 s | 1.03 tok/s | same prompt |

CUDA is the faster path on this card. The cold start is a one-time cost; the expert tier and the
pinned history are reused across runs.

### 8K / 32K / 64K results

CUDA, `COLI_CUDA=1`, 32 generated tokens per case, KV at the engine's default fp32:

| Context | Prompt tokens | Prefill | Prefill tok/s | Decode | Peak RSS | KV (fp32) | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8K | 8,027 | 50.1 s | 160.2 | 20.3 tok/s | 9.1 GB | 824 MiB | 61.7 s |
| 32K | 32,100 | 305.8 s | 105.0 | 15.0 tok/s | 11.1 GB | 2,329 MiB | 320.6 s |
| 64K | 63,977 | 1,009.4 s | 63.4 | 14.2 tok/s | 11.3 GB | 4,321 MiB | 1,020.0 s |

The 64K run used 10.9 GB of VRAM and was close to the card's limit; its prefill breakdown was
attention 736.8 s and expert MM 235.5 s (the rest fill/other). It completed, but the 16.8-minute
prefill is the reason the KV work matters. The benchmark harness uses a local `getline` patch in
the `-f` prompt reader; upstream's `fgets(8192)` would split a 64K prompt into many unrelated
prompts.

### Unsloth b12588 baseline at 64K

Unsloth b12588, IQ3_XXS + DSpark, `-c 65536`, q4 KV (`-ctk q4_0 -ctv q4_0 -fa on`),
`-b 512 -ub 128`, `--n-cpu-moe 12`. `--n-cpu-moe 8` did not fit (KV allocation OOM); 12 was the
minimum that loaded. Model load 4.0 s; GPU 10.4 GB used during the run.

| Backend | Prefill | Prefill tok/s | Decode | Decode tok/s | Notes |
|---|---:|---:|---:|---:|---|
| Colibri CUDA | 1,009.4 s | 63.4 | 2.2 s / 32 | 14.16 | fp32 KV 4.3 GiB, RSS 11.3 GB |
| Unsloth b12588 | 439.8 s | 145.5 | 1.69 s / 32 | 18.33 | q4 KV, ncmoe 12, DSpark 77.8% acceptance |

**Result: Colibri does not help at 64K on this box.** Unsloth is **2.3× faster at prefill** and
**1.29× faster at decode**, with q4 KV and DSpark. Colibri's expert cache is real, but its fp32 KV
and attention path dominate: 736.8 s of the 1,009.4 s prefill was attention (73%), while expert MM
was 235.5 s. KV16/q4 is necessary for Colibri to compete at 250K/1M, but it is not sufficient on
its own — the long-context attention/prefill path needs work too.

For 250K today, use the llama.cpp plan. For 1M, Colibri remains interesting because expert
streaming + q4 KV could fit weights/KV where llama.cpp cannot, but only after the KV and attention
work.

## 3. The KV budget

Inkling 2 Mini has 48 state layers: 24 physical plus 24 loop-pass slots. Eight of them are global
attention (layers 5, 11x3, 17x3, 23); the other 40 are sliding-window rings of 1024 tokens. The
global slots dominate long context.

Per token per global slot: `2 * 8 KV heads * 128 = 2048` values.

| KV type | Per token, 8 global slots | 32K | 64K | 250K | 1M |
|---|---:|---:|---:|---:|---:|
| fp32 (today) | 64 KB | 2.05 GB | 4.10 GB | 16.0 GB | 64.0 GB |
| fp16 (`KV16`) | 32 KB | 1.02 GB | 2.05 GB | 8.00 GB | 32.0 GB |
| q8 | 17 KB | 0.56 GB | 1.11 GB | 4.35 GB | 17.4 GB |
| q4 | 9.2 KB | 0.30 GB | 0.60 GB | 2.30 GB | 9.2 GB |

SWA rings: ~0.17 GB fp32 (constant); conv banks: ~4 MB. Dense: ~1.4 GB. The current `coli plan`
reports 5.1 / 7.5 / 10.8 GB "runtime" at 8K / 32K / 64K, which is mostly KV plus activations.

The VRAM side is as important as RAM. The Vulkan dense chain keeps a K/V mirror (`kvl`) for the
state layers. At 64K fp32 that mirror is 4.1 GB on top of the 8.2 GiB expert tier; at 250K it is
16 GB and cannot fit. The mirror must become f16/q4 or stay host-side.

## 4. Stage 1 — KV16 for 250K

**Prerequisite:** the 64K baseline shows **73% of prefill is attention** (736.8 s of 1,009.4 s),
not expert MM. KV16/q4 alone will not make 250K usable; it must come with attention/prefill work:
larger and better-shaped prompt chunks, q4/q8 attention kernels, and fewer host/device round-trips.
This stage is necessary, not sufficient.

### 4.1 Design

Add a KV element type to the Inkling2 engine:

```
INK2_KV = f32 | f16 | q8 | q4      (default f32 for oracle runs; f16 when serving)
```

- **Host cache:** `Model.K/V` and `InkSeq.K/V` store `uint16_t` rows (`kv16_from`/`kv16_to` in
  `kv16.h`) instead of `float` rows.
- **Attention:** `attention()` and `ink2_attention()` widen each fetched K/V row to a small f32
  scratch before the dot product, or use `kv16_dot_lanes` when the layout allows. The relative
  bias, sink, and softmax paths stay f32.
- **Sliding rings:** keep the same ring geometry; only the row width changes. They are tiny.
- **Device mirror:** `inkling2_chain.h`'s `kvl` buffer and the CUDA chain's mirror become f16 (or
  q4, see below). The chain's attention kernel reads the same type; a f16 mirror is half the
  device bytes and is exact for values representable in f16.
- **Persistence:** `kv_prefix.h` / `kv_persist.h` records the element type in the file header;
  an f32 `.coli_kv` is converted on resume or refused with a message. Prefix-store rows carry their
  type too.
- **Env:** `INK2_KV=f16`; `COLI_VK_KV16=1` / `COLI_CUDA_KV16=1` for the chain mirrors, following
  the existing qwen36/OLMoE pattern.

### 4.2 250K fit

| Piece | f16 host KV | q4 host KV |
|---|---:|---:|
| Global KV, 250K | 8.0 GB | 2.3 GB |
| Dense + vision tower | 1.4 GB | 1.4 GB |
| OS + other | ~8 GB | ~8 GB |
| **Subtotal** | **17.4 GB** | **11.7 GB** |
| Left for expert cache / runtime | ~5.6 GB | ~11.3 GB |
| Device mirror at 250K | 8.0 GB (too big with the tier) | 2.3 GB (fits) |

**Recommendation for 250K on this box:** f16 host KV, **q4 device mirror**, tier reduced to
~7 GB. If the q4 mirror is not ready, run `COLI_VK_CHAIN=0` (CPU attention over the f16 host
cache) or keep the mirror host-side; decode will be slower but the fit is safe.

f16 alone is not enough on the device: 8 GB mirror + 8.2 GiB tier + dense/buffers exceeds 12 GB.
That is why the device mirror is q4 from the start, or the chain is disabled at 250K.

### 4.3 Acceptance

- Loads at `CTX_MAX=262144` on 23 GB RAM and 12 GB VRAM.
- Prefill 250K completes; time recorded. Expect it to be CPU/disk-bound.
- Needle retrieval at 240K; the fact is found.
- A tool-call round trip works (Colibri's dev Mini supports tools).
- Output coherence at 8K/32K/250K; greedy text compared with `f32` where the path is exact.
- `PEAK RSS`, expert hit rate, and device/CPU split recorded.

## 5. Stage 2 — q8/q4 for 1M

### 5.1 Why q4, not f16

At 1M, f16 global KV is 32 GB and q8 is 17.4 GB; neither fits beside the model and OS on 23 GB.
q4 is 9.2 GB: with 1.4 GB dense and ~8 GB OS that is ~18.6 GB, leaving ~4.4 GB for runtime and a
small expert cache. q4 is the only viable 1M format on this host.

### 5.2 Codec

Use a 32-value block codec, like the existing KV paths:

- q8: 32 values + one f16 scale per block; decode to f32 for the dot.
- q4: 32 values + one f16 scale per block; decode to f32 or use a quantized dot.
- Keep K and V separate; K can stay q8 while V is q4 (`INK2_KV=k8v4`) if quality needs it.
- The relative-bias and sink logits are computed from weights, not KV; they stay f32.

Implement the row read/write as `ink_kv_put` / `ink_kv_get` with a type switch, and keep the
attention body independent of storage. This also lets the prefix store and `.coli_kv` save rows
without knowing the type at each call site.

### 5.3 Device policy at 1M

A full q4 device mirror (9.2 GB) plus the 8 GiB tier does not fit 12 GB. Options, in order:

1. **Host-only attention at 1M** (`COLI_VK_CHAIN=0`): KV stays in RAM, the chain's dense work still
   runs on the device per layer. Decode is slower but memory-safe.
2. **Paged mirror:** keep a recent/hot block of KV rows on the device and stream the rest per
   layer. Full attention reads every row, so this is bandwidth-bound; measure before committing.
3. **Larger card:** 24 GB makes a q4 mirror plus a ~10 GB tier comfortable; document as the
   recommended 1M configuration.

The realistic acceptance for this host is **1M with host-side q4 KV and a reduced expert tier**,
not a fully device-resident 1M.

### 5.4 1M acceptance

- Loads at `CTX_MAX=1048576` with q4 host KV and no OOM.
- Needle retrieval at ~1M; no context corruption.
- Decode recorded honestly; expect a few tok/s, not the 8K rate.
- Quality gate vs f32 at 32K on code, JSON, and retrieval.
- If the 12 GB path cannot pass, document 24 GB as the 1M minimum and keep the code path.

## 6. Implementation map

| File | Change |
|---|---|
| `c/inkling.c` | `Model.K/V`, `InkSeq.K/V` become typed; `kv_alloc` and `ink_kv_later` size by type; `attention()` reads via `ink_kv_get`; `kv_ring_rows` unchanged; `.coli_kv` load/save type tag. |
| `c/inkling2.h` | `ink2_attention()` reads typed rows; the loop-pass K/V writes go through the same put/get; no change to relative bias, sink, qk-clip, or conv math. |
| `c/inkling2_chain.h` | `kvl` mirror type (f16/q4) and its row size; chain attention reads the mirror type; `inkc_report` prints it. |
| `c/inkling2_cuchain.h` | CUDA mirror type; dequant on upload or a q4 attention kernel. |
| `c/kv16.h` | Reuse the f16 conversion and dot helpers; add f16 row get/put wrappers. |
| `c/kv_fp8.h`, `c/kv_tq.h` | Reference for q8/q4 block codecs and row layouts. |
| `c/kv_prefix.h`, `c/kv_persist.h`, `c/prefix_store.h` | Type tag in the record/file; convert or refuse on a type mismatch. |
| `docs/ENVIRONMENT.md`, `docs/inkling.md` | `INK2_KV`, `COLI_VK_KV16`, `COLI_CUDA_KV16`, `CTX_MAX`; memory tables. |
| `tests/` | f16/q8/q4 round-trip; oracle token parity for f16; quality harness for q8/q4; 250K/1M fit smoke. |

## 7. Validation

1. **Round trip:** random rows through put/get for each type; f16 exact where representable.
2. **Oracle:** `ref_inkling.json` with `INK2_KV=f16` (and q8/q4 where near-ties are acceptable);
   compare tokens and logits.
3. **Long context:** needle at 32K/128K/250K; MRCR-style multi-needle if available.
4. **Agent:** tool call round trip, prefix reuse across turns, cancellation.
5. **Memory:** `PEAK RSS`, `[kv]` line, device mirror bytes, expert hit rate.
6. **Performance:** prefill and decode at 8K/32K/64K/250K/1M; report the CPU/device split.

## 8. Risks

| Risk | Mitigation |
|---|---|
| f16/q4 KV changes near-tie tokens | Use f32 for oracle; measure KLD/top-1; keep q8 K for quality. |
| Device mirror too large at 250K | q4 mirror or `COLI_VK_CHAIN=0`; reduce tier. |
| Prefix store / `.coli_kv` type mismatch | Type tag and conversion; refuse with a clear message. |
| 1M decode is very slow | Host-side q4 KV; document; recommend 24 GB for device mirror. |
| MTP verify over quantized KV | Test with `INK2_MTP=1`; the verify rows must use the same KV type. |
| Colibri dev API churn | Pin a commit; keep the changes small and behind env flags. |
| 23 GB RAM | q4/q8, smaller expert cache, OS headroom; no swap. |
| Attention dominates long prefill | 73% at 64K; optimize chunk size/kernels before claiming 250K is usable. |

## 9. Milestones

- **M0** — Build dev + import IQ3; benchmark 8K/32K/64K.
- **M1** — f16 host KV + f16/q4 device mirror; 250K load and needle.
- **M2** — q8/q4 host KV; 250K quality/perf; 1M load test.
- **M3** — 1M acceptance or documented 24 GB requirement.
- **M4** — Upstream the KV type work behind `INK2_KV` with docs/tests.

## Appendix — benchmark commands

```bash
# build (CUDA primary; Vulkan fallback is `VK=1`)
make -C c inkling CUDA=1 CUDA_HOME=/opt/cuda

# import
./c/coli import /path/Inkling-2-Mini-UD-IQ3_XXS.gguf \
  --out /path/iq3xxs.colibri --no-mtp

# run (long prompts via -f; the local getline patch keeps one line whole)
SNAP=/path/iq3xxs.colibri CTX_MAX=65536 COLI_CUDA=1 \
  ./c/inkling -f bench/prompt-64k.txt -n 32
```

`bench/run_bench.py` runs 8K/32K/64K and writes `bench/results.json`;
`bench/unsloth_64k.py` is the llama.cpp baseline.
