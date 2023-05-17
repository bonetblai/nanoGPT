# Andrej Karpathy's nanoGPT
# Super simlpe bigram language model

import torch
import torch.nn as nn
from torch.nn import functional as F

# Reproducibility
torch.manual_seed(1337)

# Hyperparameters
g_batch_size =   32  # number of independent sequences processed in parallel
g_block_size =    8  # maximum context size for prediction
g_opt_steps  = 3000  # number of optimization steps
g_eval_freq  =  300  # frequency of evaluation
g_eval_iters =  200  # number of random batches used to evaluate loss
g_gen_length =  500  # length of generated text at the end

g_learning_rate = 1e-2
g_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download Shakespeare dataset
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt all_shakespeare.txt
with open('all_shakespeare.txt', 'r', encoding='utf-8') as fd:
    g_text = fd.read()

# Get vocabulary (i.e., characteres) in text
g_chars = sorted(list(set(g_text)))
g_vocab_size = len(g_chars)

# Create encoder/decoder of text
# [a more sofisticated word-chunk level encoder/decoder can be found at tiktoken, or SentencePiece]
g_stoi = { ch:i for i,ch in enumerate(g_chars) }
g_itos = { i:ch for i,ch in enumerate(g_chars) }
encode = lambda s: [g_stoi[ch] for ch in s]
decode = lambda l: ''.join([g_itos[i] for i in l])

# Encode entire text and store it in a pytorch tensor, split data into train and val sets
g_data = torch.tensor(encode(g_text), dtype=torch.long)
n = int(0.9 * len(g_data))
g_train_data = g_data[:n]
g_val_data = g_data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = g_train_data if split == 'train' else g_val_data
    ix = torch.randint(len(data) - g_block_size, (g_batch_size,))
    x = torch.stack([data[i:i+g_block_size] for i in ix])
    y = torch.stack([data[i+1:i+g_block_size+1] for i in ix])
    x, y = x.to(g_device), y.to(g_device)
    return x, y

# Loss
@torch.no_grad()
def estimate_loss():
    out = {}
    g_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(g_eval_iters)
        for k in range(g_eval_iters):
            X, Y = get_batch(split)
            logits, loss = g_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    g_model.train()
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
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx)
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
g_model = BigramLanguageModel(g_vocab_size)
g_m = g_model.to(g_device)

# Create pytorch optimizer
g_optimizer = torch.optim.AdamW(g_m.parameters(), lr=g_learning_rate)

# Optimize
for iter in range(g_opt_steps):
    # every once in a while evaluate model on train and val datasets
    if iter % g_eval_freq == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss and do one optimizer step
    logits, loss = g_m(xb, yb)
    g_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    g_optimizer.step()

# Generate from the model
g_context = torch.zeros((1, 1), dtype=torch.long, device=g_device)
print(decode(g_m.generate(g_context, max_new_tokens=g_gen_length)[0].tolist()))

