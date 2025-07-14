# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Authors**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby  
**Link**: [PDF](https://arxiv.org/pdf/2010.11929.pdf)

---

## Overview

This paper introduces the Vision Transformer (ViT), a novel architecture that applies a standard Transformer encoder, originally developed for NLP tasks, directly to sequences of image patches for image classification. The authors demonstrate that, when trained on sufficiently large datasets, ViT matches or surpasses the performance of state-of-the-art convolutional neural networks (CNNs) while requiring fewer computational resources for similar accuracy. By eliminating convolutions and relying entirely on self-attention, ViT highlights that much of the inductive bias in CNNs is unnecessary when ample data is available, and that global attention can capture long-range dependencies more effectively.

---

## Motivation

Convolutional neural networks have long been the dominant architecture in computer vision, owing to their strong inductive biases: locality, translation equivariance, and hierarchical feature learning. These properties make CNNs data-efficient and effective even with relatively small datasets. However, CNNs process information locally, relying on stacking many layers to capture global context, and their hardcoded assumptions may limit their flexibility on large datasets.

Transformers, which rely on self-attention mechanisms, have demonstrated remarkable success in NLP by capturing global dependencies and being highly parallelizable. Inspired by this success, the authors hypothesize that a pure Transformer could perform equally well for vision tasks if trained on sufficiently large datasets, bypassing the need for CNN-specific inductive biases.

---

## Background

CNNs process images as 2D grids of pixels through convolutional kernels that slide over local regions and progressively build hierarchical representations. They are efficient, but inherently limited to local operations.

Transformers operate on sequences of tokens. In NLP, each word or subword is represented as an embedding, and self-attention layers model relationships between all tokens at each layer. However, Transformers are permutation-invariant and do not inherently encode sequence order, so positional encodings are added to preserve order.

Applying Transformers to images requires adapting images into a sequence of tokens. ViT achieves this by splitting each image into non-overlapping square patches, flattening each into a vector, and projecting it into an embedding space. These patch embeddings are treated as tokens in the Transformer.

Because Transformers lack the built-in inductive biases of CNNs, they require larger datasets and pre-training to perform well.

---

## Vision Transformer (ViT) Architecture

ViT processes images as follows:
- The input image of size $H \times W \times C$ is divided into fixed-size patches of size $P \times P$.
- Each patch is flattened into a vector of size $P^2 \cdot C$ and mapped linearly to a $D$-dimensional embedding.
- A learnable classification token ([CLS]) is prepended to the sequence of patch embeddings. The final representation of this token after the Transformer layers is used for classification.
- Learnable positional embeddings are added to each token embedding to preserve spatial order.
- The resulting sequence is fed through a standard Transformer encoder: layers of multi-head self-attention, MLPs, residual connections, and layer normalization.

Unlike CNNs, ViT has a global receptive field at every layer through self-attention, allowing direct modeling of long-range dependencies.

---

## Training and Evaluation

ViT is pre-trained on large datasets and fine-tuned on smaller ones. The authors use:
- ImageNet-21k (14M images) and JFT-300M (300M images) for pre-training.
- ImageNet-1k, CIFAR-100, and VTAB (19 tasks) for fine-tuning and evaluation.

Training is performed using standard techniques: Adam optimizer with warmup and cosine decay learning rate schedule, label smoothing, stochastic depth, and data augmentation.

ViT comes in several sizes (Base, Large, Huge) and with different patch sizes (e.g., $16 \times 16$, $32 \times 32$). Smaller patch sizes lead to longer sequences and higher computational costs but better accuracy.

---

## Results

On ImageNet and other benchmarks, ViT matches or exceeds state-of-the-art CNNs such as ResNets and EfficientNets when pre-trained on large datasets. ViT is also more computationally efficient, achieving better accuracy per FLOP in high-data regimes. On VTAB and few-shot tasks, ViT demonstrates strong transferability, further confirming its flexibility.

Hybrid architectures, which use a CNN to generate feature maps before feeding them into a Transformer, perform better when data is limited, highlighting the tradeoff between inductive bias and flexibility.

---

## Analysis

ViT shows that global self-attention can completely replace convolutions for image classification when enough data is available. The model is less data-efficient than CNNs due to the absence of hardcoded assumptions about spatial structure, but it scales better with data size and model size. The results suggest that many of the architectural choices in CNNs were compensating for limited data rather than being fundamentally optimal.

The authors also find that ViTs produce attention maps that correlate well with object regions in images, suggesting that self-attention can implicitly learn to focus on salient parts of the input.

---

## Conclusion

ViT demonstrates that pure Transformer architectures can achieve state-of-the-art results in computer vision, challenging the dominance of CNNs. By treating images as sequences of patches and leveraging the flexibility and scalability of Transformers, ViT opens the door to unifying NLP and vision architectures under the same paradigm, provided sufficient pre-training data is available.