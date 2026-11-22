# PLAN — Strata-style long context for Inkling 2 Mini in the Unsloth llama.cpp fork

Status: draft
Target: **250K-token context (MVP)**, **1M-token context (Stage 2)**
Model: `unsloth/Inkling-2-Mini-GGUF` UD-IQ3_XXS (root Unsloth layout, `general.architecture = inkling2`) + `dspark-Inkling-2-Mini-Q8_0.gguf`
Build: this fork, based on `unslothai/llama.cpp` `b12588-mix-c0c3de8` (the build that carries the looped layout and DSpark)

## 1. Context and goal

Inkling 2 Mini is a 24-layer looped MoE: layers 6–17 run three times with shared weights, giving 48 layer applications per token from 24 stored layers. It has 128 experts per layer, top-8 routing, hybrid sliding-window/global attention (20 SWA layers with a 1,024-token window; global layers 5, 11, 17, 23), relative-position bias instead of RoPE, attention residuals, and a separate DSpark drafter. The model advertises a 1M-token context.

The current local baseline is:

- RTX 5070 **12 GB**, i5-10400F (6C/12T, AVX2), **23 GB RAM**, Linux.
- Real free VRAM with the desktop running: **~10.86 GB** (10,355 MiB); ~11.56 GB after closing Zen/Vesktop/Steam. The compositor/shell/driver overhead is ~1.4 GB and cannot be reclaimed without logging out.
- Q4_K_M (16.61 GB) on mainline llama.cpp + PR #32704, `--n-cpu-moe 24`, 8K context: **~12.4 tok/s decode, ~126 tok/s prefill**, ~10.5 GB VRAM. 32K prefill crashed; vision and tool calling were broken in that build/layout.

The goal is to move from that to a usable **250K** context on the same card with IQ3_XXS + DSpark, and then design the path to **1M**. We are not porting Strata's engine. Strata is a separate ggml-based engine with custom kernels; we are porting its **policies** into llama.cpp:

- quantized/compressed KV cache;
- reuse-aware expert placement (keep reused weights on the GPU, offload single-use ones);
- KV growth/streaming for very long context;
- prefill chunking and overlap;
- conversation checkpoints.

Not ported: PLE n-gram streaming (Inkling has no PLE), MTP/DFlash2 (Inkling uses DSpark, already in this build), QSA/GDN kernels, the Qwen vision helper, and Strata's pack/expert-arena format.

## 2. Memory model

### 2.1 KV slots

Inkling has 48 KV slots: 24 physical layers plus 24 loop-pass slots. Eight slots are global attention (layers 5, 11, 17, 23; 11 and 17 are looped three times), and 40 are sliding-window slots capped at 1,024 tokens.

Per token per global slot: `2 × 8 KV heads × 128 = 2,048` values.

| KV type | Per token, 8 global slots | 250K | 262,144 | 1M |
|---|---:|---:|---:|---:|
| fp16 | 32 KB | 8.2 GB | 8.6 GB | 34.4 GB |
| q8_0 | 17.0 KB | 4.35 GB | 4.56 GB | 18.3 GB |
| **q4_0** | **9.2 KB** | **2.30 GB** | **2.42 GB** | **9.7 GB** |
| k8v4 (K q8, V q4) | 13.3 KB | 3.33 GB | 3.49 GB | 14.0 GB |

SWA slots: ~0.17 GB (fp16) / ~0.08 GB (q4) regardless of context. Conv state: ~4 MB. DSpark: 0.40 GB weights + ~0.27 GB draft KV at 250K.

### 2.2 Weights (UD-IQ3_XXS, measured with `strata_inspect`)

- Total: **11.65 GB**, 26.07 B weights, 3.58 bits/weight.
- Routed experts: **10.30 GB** at 3.37 bits/weight; **3.35 MB per expert**; 0.429 GB per physical layer.
- Loop core (layers 6–17): **5.15 GB**; prelude/final (0–5, 18–23): **5.15 GB**.
- Dense (attention, head, embeddings, router, other): **1.36 GB**.

### 2.3 Fitting 250K

With q4 KV, DSpark, SWA and ~0.5–1.0 GB scratch:

```
KV q4          2.30–2.42 GB
SWA            0.17 GB
DSpark         0.40 GB weights + 0.27 GB draft KV
scratch        0.5–1.0 GB
overhead       ~3.6–4.1 GB
weights budget 10.86 - overhead ≈ 6.8–7.2 GB
```

Dense + loop core = **6.51 GB**, so it fits with only ~0.3–0.7 GB to spare; the 12 single-use layers' experts (5.15 GB) go to CPU/RAM. This is tight: close Zen/Vesktop/Steam, and consider `-c 250000` rather than 262,144 if OOM.

q8 KV (4.35 GB) does not leave room for the loop core with IQ3 (budget ~5.3 GB); fp16 is impossible. q4 KV is the MVP enabler. k8v4 may fit only with a slightly smaller context or one additional offloaded layer.

### 2.4 Why 1M is a different problem

At 1M, q4 global KV alone is **9.7 GB**. Weights + KV + overhead is ~15.5–16 GB, far beyond 10.86 GB. Even q8/q4 hybrid is ~14 GB. There is no placement trick that makes this fit; Stage 2 needs one or more of:

- KV streaming/eviction with a bounded VRAM window;
- block-sparse or landmark attention for the 8 global layers (quality trade);
- a substantially smaller quant (IQ1/IQ2) and still only ~1M with streaming;
- more VRAM (24–32 GB) or a different machine.

The plan treats 1M on 12 GB as a research target with explicit quality and speed gates, not a promised configuration.

## 3. MVP — 250K context

### 3.1 Deliverables

1. **Reuse-aware placement helper** (`tools/strata_placement.py` or equivalent) that reads the GGUF loop metadata and emits the correct `-ot` rules:
   - Offload `blk.{0–5,18–23}.ffn_{gate_up,down}_exps.{weight,scale}` to the CPU.
   - Keep the loop core (`blk.{6–17}`) and all dense/attention tensors on the GPU.
   - Handle both metadata prefixes: `inkling2.*` (root Unsloth layout) and `inkling2_loop.*` (Mainline layout). Derive the prefix from `general.architecture`.
2. **250K launch recipe** (see §3.5) using q4 KV, FlashAttention, DSpark and the placement rules.
3. **Benchmark/validation script** that records VRAM, prefill tok/s, decode tok/s, DSpark acceptance, and a long-context retrieval check.
4. **Docs** (`docs/inkling-long-context.md`): the memory math, the layout warning (root `inkling2` vs Mainline `inkling2_loop`), and the expected prefill cost.

### 3.2 Reuse-aware placement (the core Strata policy)

Inkling's loop makes layer-granular offload inefficient: a loop layer is used three times, so offloading it triples CPU work. The loop core is in the middle (6–17), and a naive `--n-cpu-moe N` may or may not avoid it depending on ordering. The MVP makes the policy explicit:

```
-ot 'blk\.([0-5]|1[89]|2[0-3])\.ffn_(gate_up|down)_exps\.(weight|scale)=CPU'
```

with `-ngl 99` so everything else goes to the GPU. This is the same idea as Strata's expert cache, expressed with llama.cpp's existing per-tensor override; no engine port is required.

Acceptance for this deliverable:

- `nvidia-smi` shows the loop core and dense tensors on the GPU, and only the 12 single-use layers' experts in host memory.
- A 250K load succeeds without OOM at `-c 262144` (or the documented fallback context).
- Verify the `.scale` tensors are matched for NVFP4/IQ3 layers; missing scales would corrupt output.

Later optimization: add a `--n-cpu-moe-reuse` flag (or auto-generate the rules in `common/arg.cpp`) so users do not need the helper. The helper is the MVP; the flag is a follow-up.

### 3.3 KV

- Use **q4_0 K and V** (`-ctk q4_0 -ctv q4_0`) with **FlashAttention on** (`-fa on`).
- Validate quality at 250K: needle/retrieval, a code task, and a tool-call round trip.
- If q4 quality is unacceptable, try `-ctk q8_0 -ctv q4_0` (3.33 GB at 250K). It may require reducing context to ~200K or offloading one loop layer; document the trade.
- Do not use `--swa-full`.

### 3.4 DSpark

- Use the root Unsloth GGUF and `dspark-Inkling-2-Mini-Q8_0.gguf`.
- `--spec-type draft-dspark --spec-draft-n-max 3 --n-gpu-layers-draft 99`.
- Do not use `-devd`; the drafter borrows the target's embeddings and head and must span the same devices.
- Mainline llama.cpp (even with PR #32704) does not support DSpark; this fork does.
- Note: greedy output is not guaranteed bit-identical with the drafter on CUDA (near-tie verification flips); record acceptance and any divergence.

### 3.5 Launch recipe (MVP)

```bash
./llama.cpp/llama-server \
  -m Inkling-2-Mini-UD-IQ3_XXS.gguf \
  -md dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 3 --n-gpu-layers-draft 99 \
  -ngl 99 \
  -ot 'blk\.([0-5]|1[89]|2[0-3])\.ffn_(gate_up|down)_exps\.(weight|scale)=CPU' \
  -c 262144 \
  -ctk q4_0 -ctv q4_0 -fa on \
  -b 8192 -ub 2048 \
  --jinja --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --host 127.0.0.1 --port 8082
```

Close the optional GPU apps first. If it OOMs: `-c 200000`, or move one loop layer to CPU (accepting 3× CPU work for that layer), or reduce `-ub`.

### 3.6 Prefill reality

This is the main risk. With 12 single-use layers on the CPU, the first 250K prefill is CPU-bound:

- Baseline Q4_K_M with 24 CPU layers: 126 tok/s prefill → 250K ≈ 33 min.
- IQ3_XXS with 12 CPU layers: roughly **15–30 min**, depending on cores and `-ub`.
- All-GPU experts would be ~3–5 min, but that needs ≤~6.9 GB of weights (a 1-bit quant) or more VRAM.

Strata's overlap/prefetch can hide transfers, not CPU FLOPs. The MVP must document this and rely on prompt caching/checkpoints for multi-turn agent use: the first long prompt pays, later turns read only the new tokens.

### 3.7 Validation and acceptance

- Loads at 250K with q4 KV, no OOM, VRAM ≤10.8 GB.
- Records prefill time/tok-s for 250K and decode tok/s on a 1K continuation.
- Needle retrieval at 240K and 250K: the fact is found.
- Tool-call round trip succeeds; no request poisons later requests (the failure seen on mainline + `--jinja`).
- DSpark acceptance and speed recorded with/without the drafter.
- `--parallel 1`; no cross-slot corruption.
- Output coherence: math/JSON/prose sanity at 8K and 250K.

### 3.8 Out of scope for MVP

- 1M context.
- Per-expert dynamic cache and KV streaming.
- Vision (the mtmd path produced garbage in earlier testing; separate issue).
- Quality tuning beyond the q4-KV gate.
- Mainline `inkling2_loop` layout and mainline DSpark.

## 4. Stage 2 — 1M context

### 4.1 Blockers

- q4 global KV at 1M is 9.7 GB; weights and scratch push the total to ~15–16 GB.
- Full attention over a streamed 1M KV is bandwidth-bound: reading 9.7 GB per token over PCIe is ~100–300 ms even before compute.
- The 40 SWA slots are already bounded; only the 8 global slots matter.

### 4.2 Workstreams

1. **Per-layer KV types.** SWA layers q4; global layers q8 K / q4 V. Extend `llama_kv_cache` / `llama_memory_hybrid_iswa` to accept per-layer types instead of one global `type_k`/`type_v`. Improves quality at a modest size cost.
2. **KV grow.** Allocate the global KV lazily as the context grows, giving VRAM back to experts (Strata's `--kv-grow` idea). Requires a growable or paged KV allocator rather than the current upfront `size` allocation.
3. **KV streaming / resident window.** Keep the hot/recent N cells in VRAM and the rest in pinned host memory; fetch on demand from the attention kernel. This is the largest change: attention kernels and the hybrid memory manager need a host-resident path. For full attention it is bandwidth-bound, so pair it with sparsity.
4. **Sparse/landmark global attention (optional quality trade).** If full 1M KV streaming is too slow, approximate the 8 global layers with landmarks/sinks or block selection, gated by retrieval quality. This is a model-specific research item.
5. **Per-expert cache/streaming.** Generalize the MVP placement into a residency cache: hot experts in VRAM, the rest in host RAM, optional SSD; a custom MoE op/scheduler fetches experts. Reuse-aware policy (loop core first) is the starting point. This is the biggest engineering item and should follow the KV work, because at 250K the `-ot` policy already captures most of the benefit.
6. **Prefill expert prefetch.** Overlap expert transfers with attention across layers; chunked prefill with a bounded activation window. Helps I/O, not CPU FLOPs.
7. **Conversation checkpoints.** Save/restore KV + conv state for long agent sessions; use the existing slot save/restore as the base and add Inkling's 48-slot conv state.
8. **Quant strategy for 1M.** IQ2_XXS (8.49 GB) or IQ1 (7.76 GB) reduce weights but not KV; combine with streaming. Measure quality against Q4_K_M at 32K and at the target context.
9. **DSpark at 1M.** Draft KV and acceptance at long context; reserve VRAM for the drafter and verify the 4-token window still pays.

### 4.3 Acceptance for Stage 2

- 1M load with a bounded VRAM budget (target ≤10.8 GB) using streaming/eviction, or a documented minimum of 24 GB if the 12 GB path cannot meet quality/speed.
- Needle retrieval at 1M; no context corruption.
- Decode ≥3–5 tok/s; prefill time recorded (minutes, not hours) with prompt caching.
- Quality gate vs the Q4_K_M 32K baseline on code, JSON, and retrieval; document the approximation trade.

## 5. Risks and mitigations

| Risk | Mitigation |
|---|---|
| q4 KV hurts long-context retrieval | Test q8 K/q4 V; per-layer types; needle/quality gate before calling MVP done. |
| CPU prefill dominates | Document; prompt caching/checkpoints; smaller quant; accept 15–30 min first prefill. |
| `-ot` misses `.scale` or wrong tensors | Validate regex with `--verbose`/`nvidia-smi`; include `.scale`; test output coherence. |
| Root vs Mainline layout confusion | Enforce root `inkling2` + this fork; Mainline_Support needs PR #32704 and has no DSpark. |
| Memory margins | Close desktop apps; fallback `-c 200000`; offload one loop layer if needed. |
| Fork drift vs upstream | Pin b12588; track PRs #32704 (mainline loop fix) and #32218 (DSpark). |
| DSpark near-tie flips | Document; compare with/without drafter on a fixed prompt set. |
| 1M not achievable on 12 GB | State the hardware requirement; keep the streaming path behind a flag and measure. |

## 6. Milestones

- **M0** — Fork + this plan. (done)
- **M1** — Placement helper and 250K load with q4 KV + DSpark.
- **M2** — 250K benchmark, retrieval/tool-call validation, documented prefill cost.
- **M3** — Per-layer KV types and KV-grow prototype; measure gains at 250K/500K.
- **M4** — KV streaming and/or sparse global attention prototype; 1M fit test.
- **M5** — Per-expert cache prototype (if M4 shows it is needed); 1M acceptance.

## Appendix A — tensor patterns

Offload the 12 single-use layers' expert weights and scales:

```
blk\.([0-5]|1[89]|2[0-3])\.ffn_(gate_up|down)_exps\.(weight|scale)
```

Keep on GPU: `blk.{6-17}` experts, all `attn_*`, `ffn_norm`, `ffn_gate_inp`, `exp_probs_b`, embeddings, output head, norms.

Note: tensor names are the same in both layouts; the metadata prefix differs (`inkling2.*` vs `inkling2_loop.*`). Derive the prefix from `general.architecture`.

## Appendix B — baseline measurements

| Item | Value |
|---|---|
| GPU | RTX 5070 12 GB (12,227 MiB) |
| Free VRAM, desktop loaded | 10,355 MiB (10.86 GB) |
| Free VRAM, Zen/Vesktop/Steam closed | ~11,026 MiB (11.56 GB) |
| Desktop/driver overhead | ~1.4 GB |
| Q4_K_M | 16.61 GB total, 14.78 GB experts, 4.83 bpw |
| IQ3_XXS | 11.65 GB total, 10.30 GB experts, 3.37 bpw |
| Q4_K_M 8K, `--n-cpu-moe 24` | 12.4 tok/s decode, 126 tok/s prefill, ~10.5 GB VRAM |
| 32K prefill (Q4, old build) | server died mid-prefill |
| DSpark Q8_0 drafter | 0.40 GB, 54 tensors, arch `dflash` |

## Appendix C — references

- Thinking Machines, “Introducing Inkling 2 Mini” and model card, Nov 2026.
- Unsloth, “Inkling 2 Mini — Run Locally Guide”; `unsloth/Inkling-2-Mini-GGUF` model card, `Mainline_Support/README.md`, `dspark/README.md`.
- `unslothai/llama.cpp` release `b12588-mix-c0c3de8`.
- llama.cpp PRs #32694 (Inkling 2 Mini), #32704 (looped NVFP4 scale fix), #32218 (DSpark).
- Strata docs: `HOW_IT_WORKS.md`, `DETAILS.md`, `QWEN36.md` (policies: expert cache, KV compression/streaming, prefill chunking, checkpoints).
