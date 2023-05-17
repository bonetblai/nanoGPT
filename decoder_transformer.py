# Andrej Karpathy's nanoGPT
# Simlpe decoder transformer language model

import torch
import torch.nn as nn
from torch.nn import functional as F

# Reproducibility
torch.manual_seed(1337)

# Hyperparameters
g_batch_size =   32  # number of independent sequences processed in parallel
g_block_size =    8  # maximum context size for prediction
g_opt_steps  = 5000  # number of optimization steps
g_eval_freq  =  500  # frequency of evaluation
g_eval_iters =  200  # number of random batches used to evaluate loss
g_gen_length =  500  # length of generated text at the end

g_n_embed    =   32  # number of embedding dimensions
g_n_heads    =    4  # number of heads for multi-head attention
g_n_layers   =    6  # number of blocks
g_dropout    =  0.2  # percentage for dropout

g_learning_rate = 3e-4
g_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download Shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt all_shakespeare.txt
with open('all_shakespeare.txt', 'r', encoding='utf-8') as fd:
    text = fd.read()

# Get vocabulary (i.e., characteres) in text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create encoder/decoder of text
# [a more sofisticated word-chunk level encoder/decoder can be found at tiktoken, or SentencePiece]
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Encode entire text and store it in a pytorch tensor, split data into train and val sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - g_block_size, (g_batch_size,))
    x = torch.stack([data[i:i+g_block_size] for i in ix])
    y = torch.stack([data[i+1:i+g_block_size+1] for i in ix])
    x, y = x.to(g_device), y.to(g_device)
    return x, y

# Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(g_eval_iters)
        for k in range(g_eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Scaled dot self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(g_n_embed, head_size, bias=False)
        self.query = nn.Linear(g_n_embed, head_size, bias=False)
        self.value = nn.Linear(g_n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(g_block_size, g_block_size)))
        self.dropout = nn.Dropout(g_dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([ Head(head_size) for _ in range(num_heads) ])
        self.proj = nn.Linear(g_n_embed, g_n_embed)
        self.dropout = nn.Dropout(g_dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

# MLPs
class FeedForward(nn.Module):
    """ feed forward single-layer net """
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.ReLU(),
            nn.Linear(4 * width, width),
            nn.Dropout(g_dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlockNoXAttention(nn.Module):
    """ transformer block """
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # multi-head self-attention w/ residual connections
        x = x + self.ffwd(self.ln2(x)) # feed forward net w/ residual connections
        return x

# Simple decoder-only transformer
class DecoderTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads logits for next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, g_n_embed)
        self.position_embedding_table = nn.Embedding(g_block_size, g_n_embed)
        self.blocks = nn.Sequential(*[DecoderBlockNoXAttention(g_n_embed, n_heads=g_n_heads) for _ in range(g_n_layers)])
        self.ln_f = nn.LayerNorm(g_n_embed) # final layer norm
        self.lm_head = nn.Linear(g_n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=g_device)) # (T, C)
        x = tok_emb + pos_emb    # (B, T, C)
        x = self.blocks(x)       # (B, T, C)
        x = self.ln_f(x)         # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # reshape logits and targets to calculate loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to the last g_block_size tokens
            idx_cond = idx[:, -g_block_size:]
            # get the prediction
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probability of next token
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

# Instantiate model
model = DecoderTransformer()
m = model.to(g_device)

# Create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=g_learning_rate)

# Optimize
for iter in range(g_opt_steps):
    # every once in a while evaluate model on train and val datasets
    if iter % g_eval_freq == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss and do one optimizer step
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=g_device)
print(decode(m.generate(context, max_new_tokens=g_gen_length)[0].tolist()))

