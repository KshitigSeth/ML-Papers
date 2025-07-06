# Adam: A Method for Stochastic Optimization

**Authors**: Diederik P. Kingma, Jimmy Ba  
**Link**: [PDF](https://arxiv.org/pdf/1412.6980.pdf)

---

## Overview

This paper introduces Adam (Adaptive Moment Estimation), an algorithm designed to improve the efficiency and robustness of stochastic optimization, particularly in training deep learning models. Adam combines the advantages of two widely used methods: AdaGrad, which adapts learning rates for individual parameters by accumulating the sum of squared gradients, and RMSProp, which normalizes gradients using an exponentially decaying average of squared gradients. Adam extends these approaches by incorporating momentum and adding bias correction to account for initialization effects, resulting in a simple and effective optimizer that works well across a wide range of settings.

The authors frame Adam in the context of training models on noisy, sparse, and non-stationary objectives, which are common in deep learning. They demonstrate empirically that Adam achieves fast convergence, performs well even in the presence of noisy gradients, and requires little hyperparameter tuning. They also provide a theoretical regret bound under convex assumptions, lending support to its general applicability.

---

## Background

Training neural networks involves minimizing a loss function by iteratively updating parameters using gradients. In standard stochastic gradient descent (SGD), parameters are updated by moving in the direction of the negative gradient of the loss, computed over a mini-batch of data. Although SGD is simple, it suffers from sensitivity to the learning rate and inefficiency in poorly conditioned optimization landscapes, particularly when gradients vary greatly in magnitude across parameters.

Momentum methods improve on SGD by accumulating a moving average of past gradients, which accelerates progress along directions of consistent gradient and damps oscillations in directions where gradients change sign. AdaGrad adapts learning rates per parameter by accumulating squared gradients, which works well for sparse gradients but tends to slow down too much over time. RMSProp replaces the sum of squared gradients with an exponentially weighted moving average to prevent this decay. Adam integrates these ideas into a single method.

One challenge with moving averages is that they are biased toward zero at the beginning of training, since they are initialized at zero and take time to accumulate information. Adam addresses this by applying bias correction factors to both the first moment estimate (the mean of gradients) and the second raw moment estimate (the uncentered variance of gradients), producing more accurate estimates early in training.

---

## The Adam Algorithm

Adam maintains two moving averages for each parameter: the first moment estimate, which tracks the mean of the gradients, and the second raw moment estimate, which tracks the mean of the squared gradients. At each time step $t$, given the gradient $g_t$ for a parameter, the algorithm updates the first and second moments as

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

where $\beta_1$ and $\beta_2$ are exponential decay rates for the moment estimates, typically set to 0.9 and 0.999, respectively. Because $m_t$ and $v_t$ are biased toward zero at early steps, bias-corrected estimates are computed as

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$

$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

The parameter update at each step is then given by

$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where $\alpha$ is the learning rate and $\epsilon$ is a small constant (e.g., $10^{-8}$) to prevent division by zero.

The moving averages of the gradients enable momentum-like behavior, while the per-parameter scaling by the square root of the second moment adapts learning rates to the geometry of the optimization surface. The bias correction ensures that both estimates are accurate even in early iterations.

---

## Theoretical Analysis

The authors analyze Adam under the framework of online convex optimization, where an algorithm makes a sequence of decisions and suffers loss at each step, compared to the best fixed decision in hindsight. They show that Adam achieves a regret bound of $O(\sqrt{T})$ under reasonable assumptions, where $T$ is the number of iterations. This suggests that Adam converges well even in adversarial or changing environments, providing theoretical support for its robustness.

---

## Empirical Evaluation

The paper evaluates Adam on several benchmark problems, including training convolutional neural networks on the MNIST dataset, variational autoencoders, and neural machine translation models. Across tasks, Adam achieves fast convergence and competitive or superior performance compared to SGD, AdaGrad, RMSProp, and other first-order methods. It performs particularly well in settings where gradients are sparse or noisy, such as language modeling with large vocabularies or models with dropout regularization.

Adam’s default hyperparameters $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$ work well across most tasks, reducing the need for extensive tuning. The authors also propose a variant called AdaMax, which replaces the L2 norm of the second moment with the L∞ norm, providing additional stability in some settings.

---

## Conclusion

Adam offers an effective and efficient optimization algorithm for deep learning by combining the benefits of momentum and adaptive learning rates, while correcting for bias in the moment estimates. Its simplicity, strong empirical performance, and theoretical foundation have made it one of the default choices for training neural networks.