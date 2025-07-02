# Random Forests

**Author**: Leo Breiman\
**Link**: [PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

---

## Overview

This paper introduces **Random Forests**, a powerful ensemble learning method for both classification and regression tasks. Developed by Leo Breiman, the technique builds on two core ideas: **bagging** (bootstrap aggregation) and **random feature selection**. Each model in the ensemble is a decision tree trained on a bootstrapped subset of the data, and each split in a tree considers a random subset of features.

Random Forests are designed to reduce overfitting and improve generalization by combining the outputs of multiple de-correlated trees. The paper presents theoretical insights, empirical evidence, and practical tools (like Out-of-Bag error and variable importance) that make Random Forests a widely used and highly effective method.

---

## Motivation

Traditional decision trees have high variance: small changes in the data can lead to very different trees. Bagging helps reduce this variance by averaging predictions from trees trained on different bootstrap samples, but it does not address the **correlation** between the trees, which can limit the effectiveness of the ensemble.

Random Forests introduce additional randomness by selecting a random subset of features at each split. This **de-correlates** the trees, further reducing ensemble variance and improving performance. This small modification yields substantial gains in accuracy, robustness, and interpretability.

Key goals addressed by Random Forests:

- High accuracy
- Resistance to overfitting
- Internal estimates of generalization error
- Estimations of feature importance
- Fast and scalable implementation

---

## Algorithm: How Random Forests Work

Given a training set with `n` data points and `p` features:

1. **Bootstrap Sampling**:

   - For each of `B` trees, draw a bootstrap sample (with replacement) of size `n` from the training set.

2. **Tree Construction**:

   - At each node of the tree:
     - Select a random subset of `m` features (typically `sqrt(p)` for classification or `p/3` for regression).
     - Choose the best split among these features.
   - Grow each tree fully without pruning.

3. **Prediction**:

   - **Classification**: Predict the class label by majority vote across all trees.
   - **Regression**: Predict the output by averaging the predictions of all trees.

This process yields a diverse forest of trees whose aggregate prediction is more accurate and stable than any individual tree.

---

## Theoretical Insights

### Error Rate Convergence

The generalization error of a Random Forest depends on:

- **Strength**: The accuracy of individual trees.
- **Correlation**: The similarity between tree predictions.

Increasing the number of trees `B` reduces variance but does not increase bias. Random feature selection decreases correlation, which is crucial to lowering overall error.

Breiman defines the bound:

```
Generalization Error <= rho * (1 - strength^2) / strength^2
```

Where `rho` is the average correlation between trees. Thus, even if trees are weak, reducing correlation (via feature randomness) can significantly improve ensemble performance.

---

## Internal Estimates

One of the strengths of Random Forests is their ability to provide internal diagnostics without requiring a separate validation set:

### Out-of-Bag (OOB) Error Estimate

- Each tree is trained on a bootstrap sample, so \~1/3 of the data is "left out".
- For each data point, aggregate predictions from trees where that point was OOB.
- Compute accuracy (or error) based on these predictions.
- **OOB error** provides an unbiased estimate of the generalization error.

### Variable Importance

Two primary methods:

1. **Mean Decrease in Accuracy**: Randomly permute values of a feature and measure increase in OOB error.
2. **Mean Decrease in Gini Impurity**: Track how much each feature reduces Gini impurity (or MSE for regression) across the forest.

These tools help interpret models and perform feature selection.

---

## Practical Considerations

### Hyperparameters

- `B`: Number of trees (100–1000 is usually enough)
- `m`: Number of features to consider at each split
- `n`: Size of bootstrap sample (usually equal to original training size)

### Advantages

- **Parallelizable**: Each tree can be trained independently.
- **Fast inference**: Although prediction involves multiple trees, trees are shallow and fast to evaluate.
- **Robust to noise and outliers**: High accuracy with little overfitting.
- **Handles high-dimensional data well**: Especially useful when `p >> n`.

---

## Experimental Results

The paper compares Random Forests to other methods (bagging, boosting, SVMs, etc.) on various benchmark datasets. Findings include:

- Consistently high accuracy across tasks.
- Robustness to overfitting even with a large number of trees.
- Better performance than boosting on noisy datasets.
- Strong performance without needing extensive hyperparameter tuning.