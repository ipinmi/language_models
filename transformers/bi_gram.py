import torch
import torch.nn as nn
from torch.nn import functional as Func

# Hyperparameters
torch.manual_seed(1337)
batch_size = 4  # ndependent sequences to be processed in parallel
block_size = 8  # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
max_token = 500
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

# Simple Bi-gram model


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads off the nth logits for the next token from this lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # (Batch,Time,Channel) e.g (4,8,65)
        logits = self.token_embedding_table(idx)

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
            logits, loss = self(idx)  # get the predictions

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


model = BigramLanguageModel(vocab_size)
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
