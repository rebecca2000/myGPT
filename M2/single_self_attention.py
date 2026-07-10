'''
An extremely primitive GPT implementation.
Excludes many features such as drop out, multi-head, layer norm, efficiency etc.
They will be added in future milestones.
'''

import os

import torch
from torch import nn
from torch.nn import functional as F

# --- hyperparameters ---
block_size = 16
batch_size = 512
n_embed = 16
learning_rate = 1e-3
max_steps = 5001
eval_interval = 500

torch.manual_seed(1337)  # reproducibility

# --- data ---
HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, '..', 'input.txt'), encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # len(data) - block_size so the last block can safely take a full batch_size of chars
    # (batch_size,) = make a 1D tensor containing batch_size random numbers
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stack the examples together in a single tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # batch, time, channel
        k = self.key(x) # x @ self.key -> (B,T,C) @ (C,hs) -> (B,T,hs)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        # Causal mask, do not attend to tokens in the future. Resize self.tril to have dimensions TxT
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        v = self.value(x) # (B,T,C) @ (C,hs) -> (B,T,hs)
        out = weights @ v # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx) # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (B,T,C)
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x) # (B,T,C) @ (C,vocab_size) -> (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # removes the idea of separate batches. All examples put into a single batch
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # mean negative log-likelihood
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


if __name__ == '__main__':
    stoi = {c: i for i, c in enumerate(chars)} # map char to ID
    itos = {i: c for i, c in enumerate(chars)} # ID to character mapping
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    gpt = GPT()
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=learning_rate)
    for step in range(max_steps):
        xb, yb = get_batch('train')
        logits, loss = gpt(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_interval == 0:
            print(step, loss.item())

    start = torch.zeros((1, 1), dtype=torch.long)  # seed with char 0
    out = gpt.generate(start, 500)
    print(decode(out[0].tolist()))

'''
SAMPLE OUTPUT

0 4.4908976554870605
500 2.8077592849731445
1000 2.6129560470581055
1500 2.511652946472168
2000 2.4655351638793945
2500 2.422081232070923
3000 2.4300670623779297
3500 2.4136862754821777
4000 2.4159440994262695
4500 2.4091668128967285
5000 2.3914825916290283

“Mrit; alegicorftheranung of “the tet herangy fowor vene nad we wacem bemoqund; clinto fi
anmare ato
tawy.

ho ti ine ng st thes r. Iul nto whino dafl ctit?  qut ine ide parwed Ezalr lot n.
Hul I owr
n eland he thay exspr expteas ret thery edos lasnetig be, he s p,” sopre ans,”-
? 

“_”

bele
acy watlecoutr: whangneqund. Mrant, aucacincaga pe, asas jed;  smong
pertank Darienckes, Mr ig hors of vers ofy
fethatwese, I
so bunt,” wast aso liento f rechid ablyo lit esh by of mablonghist.-fas that bou
'''