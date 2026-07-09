A toy project to implement a transformer model in Pytorch, for my learning.

# Build-a-Transformer: A Milestone-Based Project

## Overview

You will implement a decoder-only transformer (a "GPT") from scratch and use it
to investigate a research question. The architecture follows the standard
decoder-only transformer (cf. Karpathy's nanoGPT); **your contribution is the
arithmetic investigation and ablation study built on top of it.**

The goal is **not** great generated text. The goal is that you can explain every
tensor shape and every design decision. Optimise for understanding.

- **Language/stack:** Python + PyTorch
- **Compute:** laptop CPU or a single small GPU is sufficient
- **Estimated effort:** ~3–5 days
- **Deliverable:** a git repo with code, plots/tables, and a
  Question → Method → Results → Learnings README.

---

## Milestone 1 — Bigram baseline & training harness

Build a lookup-table bigram model (`nn.Embedding(vocab, vocab)`) with no
attention. Purpose: get the training loop, cross-entropy loss, and sampling
working before attention exists, so later milestones have a baseline to beat.

**Corpus:** `input.txt` is Pride & Prejudice (character-level, `vocab_size = 91`).
Any single plain-text corpus works; the code is corpus-agnostic.

**What was built**
- `bigram.ipynb` — the neural `BigramLM`: `token_emb = nn.Embedding(vocab, vocab)`,
  `forward` returns `(logits, loss)` with `loss = F.cross_entropy` over `(B*T, V)`,
  an AdamW training loop, and a `generate` sampling loop.
- `bigram_lookup_table.py` — a count-based sibling that builds the same
  `P(next | current)` table by **counting** bigrams (vectorized `index_put_`),
  with Laplace (+1) smoothing, an `avg_nll` scorer, text generation, and a
  `plot_heatmap()` of `P`.

**Success criteria**
- [x] Data loads and encodes/decodes losslessly (`encode`/`decode` round-trip).
- [x] Training loop runs and training loss **decreases** (≈5.4 → ≈2.2 nats).
- [x] Model can sample text of arbitrary length without crashing.
- [x] Baseline loss recorded: converges to the **bigram floor ≈2.45 nats**
  (= conditional entropy `H(next | current)`); the neural and count-based
  models agree, confirming both reach the optimal bigram table.

**Notes / learnings**
- Use a larger batch and/or an averaged `estimate_loss` for a clean loss curve.
- `plot_heatmap()` visualizes `P(next | current)` (e.g. bright `q→u`,
  sentence-enders `.?!;` → newline), making the learned bigram structure legible.
- `generate` recomputes logits for the whole growing context each step and keeps
  only the last position — an O(N²) inefficiency a KV cache would fix later.

---

## Milestone 2 — Single-head self-attention

Add one causal self-attention head implementing Q/K/V, with a mask so position
*t* cannot attend to *t+1*. This is the conceptual core.

**Success criteria**
- [ ] Attention weights form a lower-triangular matrix (causality verified in a test).
- [ ] Attention rows sum to 1 (softmax verified).
- [ ] Validation loss is **strictly lower** than the Milestone 1 baseline.
- [ ] You can state the shape of Q, K, V, and the attention matrix from memory.

---

## Milestone 3 — Multi-head attention + feed-forward block

Run multiple heads in parallel, concatenate, then add a per-position MLP to form
one complete transformer block.

**Success criteria**
- [ ] Output shape after multi-head attention equals the input shape.
- [ ] `n_heads` is configurable and `head_size * n_heads == n_embd` holds.
- [ ] Validation loss is **lower** than Milestone 2.
- [ ] Unit test confirms one block preserves `(batch, time, n_embd)` shape.

---

## Milestone 4 — Stack blocks + residuals + layernorm

Stack multiple blocks and add residual connections and layer normalisation so a
deep model actually converges.

**Success criteria**
- [ ] `n_layers` is configurable; model builds for `n_layers >= 4`.
- [ ] Residual connections and layernorm are present in each block.
- [ ] A deep model **without** residuals/layernorm demonstrably trains worse
  (quick comparison recorded).
- [ ] Validation loss is **lower** than Milestone 3.

---

## Milestone 5 — Positional embeddings, scale up, generate

Add positional embeddings (explain why attention is otherwise order-blind), tune
hyperparameters, train longer, and generate samples.

**Success criteria**
- [ ] Removing positional embeddings measurably worsens loss (recorded).
- [ ] Final validation loss is your best so far.
- [ ] Generated samples resemble the training corpus's structure.
- [ ] Hyperparameters and final loss curve are saved to the repo.

---

## Milestone 6 — Portfolio centerpiece: arithmetic task

Retarget the **same architecture** to multi-digit addition (e.g. 3-digit `a+b`).
Generate a synthetic dataset, train, and measure **exact-match accuracy**.
Investigate the left-to-right failure mode and test the **digit-reversal** fix.

**Success criteria**
- [ ] Dataset generator produces correct, deduplicated train/test splits with no leakage.
- [ ] Exact-match accuracy is reported on a held-out test set.
- [ ] Standard vs. digit-reversed input ordering are compared with a results table.
- [ ] README explains the observed failure mode and why reversal helps.

---

## Milestone 7 — Ablation study

Ablate one component at a time (causal mask, positional embeddings, layernorm,
residuals) and quantify the impact.

**Success criteria**
- [ ] A results table compares full model vs. each ablation on the same metric.
- [ ] Each ablation is a single, isolated change (config flag, not a fork).
- [ ] Loss curves for all variants are plotted on shared axes.
- [ ] README states, per ablation, what broke and why.

---

## Final Deliverable Checklist

- [ ] Reproducible: a single documented command trains each milestone.
- [ ] Fixed random seed; results are reproducible run-to-run.
- [ ] README follows **Question → Method → Results → Learnings**.
- [ ] Prior art (nanoGPT) is cited with a one-line statement of your delta.
- [ ] Plots/tables for the arithmetic task and ablation study are included.
