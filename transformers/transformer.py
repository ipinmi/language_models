import torch
import torch.nn as nn
from torch.nn import functional as Func

# Hyperparameters
torch.manual_seed(1337)
batch_size = 64  # ndependent sequences to be processed in parallel
block_size = 256  # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_token = 500
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# Using the tiny shakespeare dataset
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("../data/input.txt", "r", encoding="utf-8") as f:
    texts = f.read()

# Unique characters in the text
chars = sorted(list(set(texts)))
vocab_size = len(chars)

# Mapping from characters to integers and back
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder:performs stoi, take a string and output a list of integers


def encode(s): return [stoi[char] for char in s]

# Decoder: performs itos, take a list of integers and output a string


def decode(l): return "".join([itos[i] for i in l])


# Train and Validation data splits
# Encoding the entire dataset and store into a Pytorch Tensor
data = torch.tensor(encode(texts), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, and the rest validation
train_data = data[:n]
val_data = data[n:]

#  Data Loading


def get_batch(split):
    """
    Generates a small batch of data of inputs x and targets y
    Args:
        - split(str): train or validation data selection
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # 4 random numbers between 0 and len(data) - block_size
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y

# Loss Evaluation


@torch.no_grad()
def estimate_loss():
    """
    Gets the average loss over multiple batches
    """
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


class Head(nn.Module):
    """ Implements a single head of self-attention (Scaled dot product attention) """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args: x(vector), input of size (batch, time-step, channels)
        Returns: output vector of size (batch, time-step, head size)
        """

        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities") and normalized with scaled attention
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # decoder block
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        wei = Func.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of concatenated self-attention in parallel  """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple MLP with a linear layer followed by a non-linearity """

    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.feedfwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # adding skip connections and layer normalization
        x = x + self.sa(self.ln1(x))
        x = x + self.feedfwd(self.ln2(x))
        return x


# Simple Generative Pre-trained Transformer
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads off the nth logits for the next token from this lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # adding tarnsformer blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        # 4 heads of 8-dim self-attention
        # self.sa_heads = MultiHeadAttention(4, n_embd//4)
        # self.feedfwd = FeedFoward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # (Batch,Time,Channel) e.g (4,8,65)
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        x = token_emb + pos_emb  # (B,T,C)
        # x = self.sa_heads(x)  # applying one head of attention (B,T,C)
        # x = self.feedfwd(x)  # (B, T, C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # cross entropy expects B,C,T
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = Func.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Extends the current context by increase the length from( B, T) to (B, T+n)
        Args:
            - idx: the current context of characters in a batch
            - max_new_tokens: the number of characters to extend by
        """
        for _ in range(max_new_tokens):

            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)  # get the predictions

            logits = logits[
                :, -1, :
            ]  # focusing only on the last time step to becomes (B, C)

            # apply softmax to get probabilities
            probs = Func.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = GPTLanguageModel()
m = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate texts from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=max_token)[0].tolist()))
