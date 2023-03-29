## Implementation of a Transformer Architecture

This contains implementations of the architecture of the transformer following the paper ["Attention Is All You Need" by Vaswani et al.](https://arxiv.org/abs/1706.03762)

---

**Transformer Model Architecture**

- PyTorch Implementation of the autoregressive neural network contains the following:

  - Decoder stack
  - Scaled Dot-Product Attention Layer
  - Multi-Head Attention Layer: consisting of several concantenated attention layers running in parallel.
  - Feed-Forward Networks Layer
  - Positional Encoding
  - Transformer blocks: communication followed by computatio
  - Add + Norm Layer: Skip connections and Layer normalization
  - Optimizer
  - Regularization : Dropout

- [Python File](transformers/transformer.py)

---
