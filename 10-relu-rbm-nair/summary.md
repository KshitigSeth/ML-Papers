# Rectified Linear Units Improve Restricted Boltzmann Machines

**Authors**: Vinod Nair, Geoffrey E. Hinton  
**Link**: [PDF](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)

---

## Overview

This paper explores the benefits of replacing traditional binary stochastic units in Restricted Boltzmann Machines (RBMs) with rectified linear units (ReLUs). The key idea is that rectified linear hidden units, which compute the function $f(x) = \max(0, x)$, can lead to better learning and generalization than traditional sigmoid units. The paper shows empirically that ReLUs lead to significantly improved performance on several benchmark datasets and also support a probabilistic interpretation through noisy sampling. The contribution is situated in the context of pretraining deep neural networks and learning generative models.

## Background and Motivation

Restricted Boltzmann Machines are energy-based generative models used to model the joint distribution between a set of visible units \( v \) and hidden units \( h \). They are a simplified form of Boltzmann Machines with no intra-layer connections, which makes inference and training tractable. Training is typically done using an approximate method called Contrastive Divergence.

Traditionally, RBMs use binary stochastic hidden units with sigmoid activation functions:

$$
P(h_j = 1 \mid v) = \sigma\left(b_j + \sum_i v_i w_{ij}\right)
$$

However, sigmoid units suffer from saturation at large magnitudes of input, leading to vanishing gradients and slow learning, particularly in deeper networks. Rectified Linear Units offer a solution: they do not saturate in the positive domain and yield sparse and high-gradient representations, which can improve learning dynamics.

The authors aim to leverage the advantages of ReLUs within the probabilistic framework of RBMs and examine whether this change improves generative and discriminative performance.

## Rectified Linear Units in RBMs

ReLUs are defined by the activation function:

$$
f(x) = \max(0, x)
$$

This function has two key properties: it does not saturate for positive inputs and it encourages sparsity by outputting zero for negative inputs. In deterministic feedforward networks, ReLUs have already shown improved optimization and generalization performance. The authors extend ReLUs to probabilistic models by interpreting them as an infinite sum of binary units with shared weights but different biases. In other words, each ReLU can be seen as the soft maximum over multiple binary units with increasing activation thresholds.

To incorporate ReLUs into RBMs, the authors define a noisy version where each hidden unit samples from a truncated normal distribution. Specifically, the activation \( h_j \) is sampled from:

$$
\text{ReLU}(a_j + z_j), \quad z_j \sim \mathcal{N}(0, 1)
$$

This makes each hidden unit a continuous, non-negative variable with stochasticity arising from Gaussian noise. Importantly, this allows the model to retain its probabilistic generative capabilities.

## Learning with ReLU RBMs

The learning procedure remains largely the same as in traditional RBMs. The model is trained using Contrastive Divergence, which approximates the gradient of the log-likelihood. The gradient of the expected energy function with respect to the weights is:

$$
\frac{\partial \log P(v)}{\partial w_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}
$$

The expectations are approximated using Gibbs sampling. Despite the continuous nature of ReLU activations, sampling remains efficient and training converges reliably.

The authors also incorporate standard regularization techniques such as weight decay and momentum, and note that ReLU RBMs benefit from sparse activation, which acts as an implicit regularizer.

## Experimental Results

The authors evaluate their model on three benchmark datasets:

1. **MNIST**: The ReLU RBM achieves lower reconstruction error and better classification performance when used to initialize a feedforward neural network. Adding Gaussian noise during training improves performance further.

2. **NORB**: Using a deeper network with ReLU RBMs leads to better generalization on this challenging 3D object classification task. The ReLU units capture more expressive features compared to binary units.

3. **LFW (Labeled Faces in the Wild)**: Using a Siamese network architecture with cosine similarity between face embeddings, ReLU RBMs achieve better verification performance than their sigmoid counterparts.

Across all tasks, ReLU RBMs outperform traditional sigmoid RBMs, both as generative models and as unsupervised pretraining components for discriminative models.

## Interpretations and Insights

One interpretation provided in the paper is that ReLU units effectively implement a mixture of linear models. Since only a subset of units are active for any given input, each input is processed by a different linear sub-model. This improves the representational capacity and learning stability.

The stochastic ReLU formulation, where Gaussian noise is added to the input before thresholding, provides a principled probabilistic framework for ReLU activations. This makes it possible to sample and estimate likelihoods in generative models, thus preserving the probabilistic nature of RBMs while reaping the benefits of ReLUs.

The improved performance is attributed to several factors:
- Higher gradient flow through non-saturating activations
- Implicit sparsity in representation
- Increased representational power via piecewise linear functions
- More effective greedy layer-wise pretraining for deep networks

## Conclusion

This paper demonstrates that replacing binary stochastic hidden units with noisy rectified linear units significantly improves the performance of Restricted Boltzmann Machines. The benefits span both generative modeling and initialization for deep discriminative models. The probabilistic formulation for ReLUs aligns well with the RBM training framework, and the empirical results support their use as a powerful alternative to traditional sigmoidal units in unsupervised learning.

The work helped pave the way for the widespread adoption of ReLUs in deep learning architectures and represents a significant milestone in bridging probabilistic models with practical deep neural networks.

