"""Count-based bigram language model.

Builds the bigram model directly by counting how often each character
follows each other character. Row i of the table is the next-char distribution
given current char i -- i.e. P(next | current).

This is the closed-form / "lookup table" sibling of the neural BigramLM: for a
pure bigram, counting recovers the same distribution the network converges to.
"""

import torch

# --- Vocab (same conventions) -----------------------------
text = open('input.txt').read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}   # char -> ID
itos = {i: c for i, c in enumerate(chars)}   # ID -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

# --- Build the bigram count table -------------------------
data = torch.tensor(encode(text), dtype=torch.long)

# N[i, j] = number of times char j immediately follows char i
N = torch.zeros((vocab_size, vocab_size), dtype=torch.float)
# current chars = data[:-1], next chars = data[1:]
cur, nxt = data[:-1], data[1:]
# scatter-add 1 into (cur, nxt) for every adjacent pair (vectorized count)
N.index_put_((cur, nxt), torch.ones_like(cur, dtype=torch.float), accumulate=True)

# --- Convert counts -> probabilities ----------------------
# +1 Laplace smoothing so no transition has probability 0 (avoids dead ends
# and log(0) if you ever compute loss).
P = (N + 1)
P = P / P.sum(dim=1, keepdim=True)   # each row sums to 1


def generate(idx, max_new_tokens):
    """Autoregressively sample characters from the count-based table.

    idx: LongTensor of shape (1, T) holding the running context.
    Only the last character matters for a bigram model.
    """
    for _ in range(max_new_tokens):
        last = idx[0, -1].item()             # current character ID
        probs = P[last]                      # P(next | current), shape (vocab_size,)
        next_id = torch.multinomial(probs, num_samples=1)  # sample one char, shape (1,)
        next_id = next_id.view(1, 1) # shape (1,1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def avg_nll():
    """Average negative log-likelihood (nats/char) of the data under P.
    Comparable to the cross-entropy loss printed by the neural model.
    """
    logprobs = torch.log(P)
    return -logprobs[cur, nxt].mean().item()


def plot_heatmap(save_path=None):
    """Visualize P as a heatmap of P(next | current).

    Row i (y-axis) = current char, column j (x-axis) = next char.
    A bright cell (i, j) means "char j often follows char i".
    """
    import matplotlib.pyplot as plt

    # Readable labels: show whitespace via repr (e.g. '\n', ' ' -> "' '").
    labels = [repr(itos[i])[1:-1] if itos[i].strip() else repr(itos[i])
              for i in range(vocab_size)]

    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(P, cmap='Blues')

    ticks = range(vocab_size)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # Put the "next character" axis on top (standard for transition matrices)
    # and make sure the tick labels are actually visible.
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', which='both', length=2, labelbottom=False, labeltop=True)
    ax.tick_params(axis='y', which='both', length=2, labelleft=True)

    ax.set_xlabel("next character")
    ax.set_ylabel("current character")
    ax.set_title("Bigram model: P(next | current)", pad=30)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"saved heatmap to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    print(f"vocab_size = {vocab_size}")
    print(f"avg NLL (nats/char) = {avg_nll():.4f}")
    print("-" * 40)

    start = torch.zeros((1, 1), dtype=torch.long)  # seed with char 0
    out = generate(start, 500)
    print(decode(out[0].tolist()))

    plot_heatmap()  # pass e.g. plot_heatmap('bigram_heatmap.png') to save to file
