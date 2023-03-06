## Implementation of Neural networks

The implementations contains a bi-gram language model, multilayer perceptron (MLP) character-level language model with activation functions, batch normalization and backprogation implemented progressively, and a WaveNet architecture model.

---

**Bi-gram character-level language model**

Implementation of a bigram character-level language model with PyTorch. The neural network contains a single linear layer followed by a softmax activation function layer. The classification loss evaluation is done using the negative log likelihood. The loss is minimized by computing the gradient of the loss wrt the weight matrices.

- [Notebook](MLP/bi-gram_nn.ipynb)

---

**Multilayer perceptron (MLP) character-level language model**

Implementation of a MLP following the paper **A Neural Probabilistic Language Model - Bengio et al. (2003) https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf**

- The [Notebook](MLP/mlp_implementation.ipynb) is a multilayer perceptron (MLP) character-level language model implemented with:
  - train/dev/test splits
  - model training
  - learning rate tuning
  - hyperparameter optimization
  - evaluation
  - under/overfitting
  - model sampling

---

**MLP with batch normalization**

Implementation of batch normalization into a two layer MLP and a six layer MLP, following the paper: **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift https://arxiv.org/abs/1502.03167**

- The [Notebook](MLP/batch_norm.ipynb) and [Notebook](MLP/NN-with-batchnorm.ipynb) implements:
  - proper scaling the weights and biases
  - introduction of batch normalization parameters such as gain, bias, running mean and variance
  - Standardizing the neurons and their firing rates ONLY at initialization
  - Visualization of the Activation distribution, Weights gradient distibution, Gradient Updates to data ratios for each layer.

---

**Manual Implementation of Backpropagation**

Implementation of the calculations/derivatives of the gradients for each parameter used in backpropagation similar to the results of **loss.backward() from PyTorch**. And backpropagating through exactly all of the variables as they are defined in the forward pass.

- [Notebook](MLP/back_propagation.ipynb)
