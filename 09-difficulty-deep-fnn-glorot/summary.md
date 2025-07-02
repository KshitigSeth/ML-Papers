# Understanding the Difficulty of Training Deep Feedforward Neural Networks

**Authors**: Xavier Glorot and Yoshua Bengio  
**Link**: [PDF](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

---

## Overview

This paper investigates why training deep feedforward neural networks has historically been difficult, and proposes a theoretically grounded solution. Prior to the deep learning revolution, deeper neural networks often failed to outperform shallower ones due to optimization difficulties. The authors argue that a core reason behind this challenge is the poor propagation of signals (both forward and backward) through many layers. Specifically, they identify improper weight initialization and activation function saturation as primary culprits that lead to vanishing or exploding gradients.

They propose a new initialization scheme, now known as Xavier (or Glorot) initialization, which helps preserve the variance of activations and gradients across layers. The paper empirically demonstrates that this initialization significantly improves the trainability of deep feedforward networks, particularly when using activation functions like the sigmoid or tanh.

## Motivation

Deep neural networks are theoretically capable of representing highly complex functions with fewer parameters than shallow networks. Yet, in practice, deeper models often underperformed or failed to train altogether. This inconsistency between theoretical expressiveness and practical performance puzzled the research community for years.

The authors point out that while tricks like unsupervised pre-training using Restricted Boltzmann Machines had shown empirical success, they lacked a clear theoretical explanation for their benefit. This paper seeks to understand and address the root cause of the optimization difficulties in deep networks.

## Background Concepts

### Gradient Flow and Saturation

In backpropagation, gradients are computed recursively through layers using the chain rule. When gradients are passed through many layers, their values can either shrink (vanish) or grow (explode) exponentially, depending on the magnitudes of weights and the derivatives of activation functions. Saturating activation functions like the sigmoid or tanh exacerbate the vanishing gradient problem since their derivatives tend to zero for large input magnitudes.

### Variance of Activations and Gradients

The authors emphasize the importance of preserving the variance of both activations and gradients across layers. If the variance of activations shrinks across layers, it results in signal attenuation and leads to inactive neurons. If it grows, it leads to unstable outputs and gradients. Maintaining consistent variance helps ensure that every layer remains in a regime where learning is possible.

### Loss Functions

The paper discusses the importance of using the right loss function. Cross-entropy loss provides stronger gradients when used with sigmoid or softmax outputs, compared to the quadratic loss. This can be crucial for training deep networks, as it ensures better gradient signals even when predictions are confident.

## Proposed Initialization Scheme

The authors derive a principled approach for initializing weights in a neural network such that the variance of activations remains constant across layers. For a layer with `n_in` input units and `n_out` output units, the weights are drawn from a uniform distribution:

$$
W \sim U\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]
$$

This distribution ensures that the expected variance of activations and gradients remains stable throughout the network, preventing early saturation or explosion. This scheme, known as Xavier initialization, is derived based on maintaining equal variance in both the forward and backward passes.

The initialization assumes that the inputs are zero-mean and that the weights and inputs are independent. These are simplifying assumptions but empirically shown to be effective.

## Activation Functions

The paper evaluates different activation functions and their impact on training deep networks:

- Sigmoid: Squashes input to the range (0, 1). Prone to saturation and vanishing gradients.
- Tanh: Zero-centered output in the range (-1, 1). Preferred over sigmoid due to better gradient flow.
- Softsign: A smoother alternative to tanh with polynomial tails. The authors find it slightly better in deep settings.

The study confirms that non-linearities with gentler saturation behavior tend to perform better, especially when paired with good initialization.

## Empirical Results

The authors conduct experiments on benchmark datasets like MNIST and deep architectures with varying depths and widths. They test combinations of:

- Initialization schemes (standard Gaussian, uniform, and the proposed Xavier scheme)
- Activation functions (sigmoid, tanh, softsign)
- Loss functions (cross-entropy vs. squared error)

The results consistently show that networks using Xavier initialization with tanh or softsign activations, trained using cross-entropy loss, outperform other configurations. In particular, deeper networks benefit the most, showing faster convergence and better generalization.

Without proper initialization, deeper models frequently underperform shallower ones. With the proposed initialization, depth becomes a strength rather than a liability.

## Pre-training as a Historical Workaround

The authors revisit why unsupervised pre-training had historically helped train deep networks. Their findings suggest that such methods implicitly brought the network weights to a favorable regime where signal propagation was not too weak or too strong. The Xavier initialization achieves a similar effect in a more direct and theoretically grounded way.

## Conclusion

This paper identifies and addresses a central problem in training deep feedforward networks: the poor propagation of signals due to ill-conditioned weight initialization. By introducing a principled method to initialize weights that preserves variance through depth, the authors provide a straightforward solution to a problem that had long hindered progress in deep learning.

The insights from this paper laid foundational groundwork for the resurgence of deep neural networks and influenced later innovations such as ReLU activations and batch normalization, which further stabilized deep architectures.

