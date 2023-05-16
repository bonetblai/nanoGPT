# Andrej Karpathy's nanoGPT
# Super simlpe bigram language model

import torch
import torch.nn as nn
from torch.nn import functional as F

# Reproducibility
torch.manual_seed(1337)

# Hyperparameters
batch_size =   32  # number of independent sequences processed in parallel
block_size =    8  # maximum context size for prediction
opt_steps  = 3000  # number of optimization steps
eval_freq  =  300  # frequency of evaluation
eval_iters =  200  # number of random batches used to evaluate loss
gen_length =  500  # length of generated text at the end

learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads logits for next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) tensor
        
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
        # idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probability of next token
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to running sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B,T+1)
        return idx

# Instantiate model
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Optimize
for iter in range(opt_steps):
    # every once in a while evaluate model on train and val datasets
    if iter % eval_freq == 0:
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
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=gen_length)[0].tolist()))

