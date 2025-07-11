# Attention is All You Need

**Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**Link**: [PDF](https://arxiv.org/pdf/1706.03762.pdf)

---

## Overview

This paper introduces the Transformer, a novel neural network architecture for sequence transduction tasks such as machine translation. Unlike previous approaches based on recurrent or convolutional networks, the Transformer relies entirely on attention mechanisms to model dependencies between sequence elements, eliminating the need for recurrence or convolutions. This innovation enables fully parallel computation over sequence elements, improving efficiency and scalability, while achieving superior accuracy on standard benchmarks. The architecture is conceptually simple yet highly effective, and it laid the foundation for many subsequent advancements in natural language processing and deep learning.

---

## Motivation

Prior to the Transformer, sequence-to-sequence models typically used encoder-decoder frameworks based on recurrent neural networks, often augmented with convolutional layers or attention mechanisms. While effective, these models suffered from several drawbacks. Recurrent computations are inherently sequential, preventing parallelization and resulting in slow training times, especially for long sequences. They also struggled to capture long-range dependencies due to vanishing gradients. Convolutional models improved parallelization but required many layers to capture distant relationships, which increased model depth and computational cost.  

Attention mechanisms, first introduced as a complement to RNNs, showed that directly modeling interactions between all input and output tokens improved the handling of long-range dependencies. The Transformer’s central insight is to eliminate recurrence and convolutions entirely, relying solely on stacked self-attention and feed-forward layers to model sequences more efficiently and effectively.

---

## Background

In sequence-to-sequence learning, the goal is to map an input sequence to an output sequence, typically using an encoder-decoder architecture. The encoder processes the input into a continuous representation, and the decoder generates the output conditioned on this representation and previously generated tokens. RNN-based models process sequences step by step, which limits parallelism and makes training slow. Convolutional models are more parallelizable but limited in their ability to model global dependencies efficiently.

Attention mechanisms allow each token to dynamically focus on relevant parts of the input, making it easier to model long-distance relationships. The Transformer generalizes this idea into a fully parallelizable architecture where every token attends to every other token in each layer.

---

## The Transformer Architecture

The Transformer is a stack of encoder and decoder layers. Each encoder layer consists of two sublayers: a multi-head self-attention mechanism and a position-wise feed-forward network. Each decoder layer has an additional encoder-decoder attention sublayer, which allows the decoder to attend to the encoder’s output. Residual connections and layer normalization wrap each sublayer, improving gradient flow and training stability.

### Self-Attention

Self-attention allows each token in a sequence to attend to all other tokens simultaneously. Given queries $Q$, keys $K$, and values $V$, the attention mechanism computes

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

where $d_k$ is the dimension of the key vectors, and the scaling by $\sqrt{d_k}$ stabilizes gradients when $d_k$ is large. The result is a weighted sum of the value vectors, with weights determined by the similarity between queries and keys.

### Multi-Head Attention

Instead of computing a single attention output, the model projects queries, keys, and values into multiple subspaces, computes attention in each subspace (head), and concatenates the results. This allows the model to capture diverse types of relationships between tokens.

### Positional Encoding

Because the Transformer lacks recurrence or convolutions, it needs explicit position information. Positional encodings are added to the input embeddings to inject sequence order information. The paper uses sinusoidal functions of varying frequencies so that the model can easily learn to attend by relative or absolute positions.

### Feed-Forward Networks

Each position in the sequence is processed by the same two-layer feed-forward network, applied identically and independently at each position.

### Residual Connections and Layer Normalization

Each sublayer is followed by a residual connection that adds the input of the sublayer to its output, and then by layer normalization. These techniques improve training stability and convergence.

---

## Training and Regularization

The model is trained with the Adam optimizer and a learning rate schedule with warmup steps followed by inverse square root decay. Dropout is applied to attention weights and feed-forward layers for regularization. Label smoothing is used in the loss function to prevent the model from becoming overconfident, which improves generalization.

---

## Results

The Transformer was evaluated on machine translation tasks: English-to-German and English-to-French. It achieved state-of-the-art BLEU scores while being significantly faster to train than comparable recurrent or convolutional models. On the English-to-German task, it reduced training costs by a factor of 4.5 compared to the previous best models. The architecture also scaled well to larger datasets and deeper networks, demonstrating its flexibility and robustness.

---

## Analysis

The Transformer’s success comes from several factors. Self-attention efficiently captures global dependencies, and its fully parallelizable structure enables faster training. Multi-head attention allows the model to jointly attend to information from different representation subspaces. The combination of attention, residual connections, and normalization stabilizes learning, while positional encodings preserve sequence order without adding sequential computation. The model’s simplicity and efficiency make it easy to scale, and its empirical success showed that recurrence is not necessary for sequence modeling.

---

## Conclusion

The Transformer fundamentally changed the way sequence transduction tasks are approached by demonstrating that attention mechanisms alone can outperform more complex architectures based on recurrence and convolution. Its parallelizable design, high accuracy, and efficiency established it as the foundation for a new generation of models in NLP and beyond, inspiring widely used architectures such as BERT, GPT, and T5.