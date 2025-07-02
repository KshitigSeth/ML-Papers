# Bagging Predictors  
**Author**: Leo Breiman  
**Link**: [PDF](https://www.stat.berkeley.edu/~breiman/bagging.pdf)

---

## Overview

This paper introduces **Bagging** (Bootstrap Aggregating), a simple yet powerful ensemble technique designed to improve the stability and accuracy of machine learning algorithms. It does so by generating multiple versions of a predictor and using them to produce an aggregated prediction. The method is particularly effective for **unstable models**—those whose predictions can change significantly with small changes to the training data.

Leo Breiman, the author of Random Forests, lays the foundation for ensemble learning here, showing that Bagging can significantly reduce variance without increasing bias, and improve performance for both classification and regression problems.

---

## Motivation

Many machine learning models suffer from **high variance**, meaning that small changes in the training data can lead to vastly different models. This is especially true for methods like decision trees, neural networks, or rule-based learners. High variance leads to **overfitting**, where the model performs well on the training set but poorly on unseen data.

The goal of Bagging is to **reduce variance** by stabilizing the output of these unstable models. The intuition is: if you train multiple models on slightly different versions of the data and then average their predictions (or take a vote), you can "cancel out" their individual noise.

Bagging achieves this using a statistical resampling technique called the **bootstrap**, which creates multiple datasets by sampling with replacement from the original training set.

---

## How Bagging Works

1. Given a dataset of size *N*, generate *B* new training datasets (typically B ~ 100) by **sampling with replacement** from the original data.
2. Train a **base learner** (e.g., decision tree) independently on each of these bootstrapped datasets to obtain predictors `f1(x), f2(x), ..., fB(x)`.
3. For a new input *x*:
   - **Regression**: Bagging prediction = average of outputs:
     ```
     f_bag(x) = (1/B) * sum_{b=1 to B} f_b(x)
     ```
   - **Classification**: Bagging prediction = majority vote among the classifiers:
     ```
     H_bag(x) = mode{f_1(x), f_2(x), ..., f_B(x)}
     ```

Each model sees a slightly different dataset and learns slightly different decision boundaries. The final ensemble aggregates them, smoothing out idiosyncrasies of any single model.

---

## Key Concepts Explained

### Bootstrap Sampling
Bootstrap sampling involves creating new datasets of the same size as the original by **sampling with replacement**. Each bootstrap dataset contains approximately 63% unique instances (some examples are repeated, some omitted). This simple perturbation is enough to produce diverse models when used with unstable learners.

### Unstable vs Stable Learners
- **Unstable learners** (like decision trees) benefit greatly from Bagging because their predictions vary significantly with small data changes.
- **Stable learners** (like k-NN or linear regression) tend not to benefit as much because their outputs are already robust to data perturbation.

### Bias-Variance Tradeoff
Bagging mainly targets **variance reduction**. It doesn't significantly affect bias (i.e., how far predictions are from the true value on average), but by reducing variability between models, it often leads to better generalization.

---

## Experimental Findings

Breiman tested Bagging on a range of algorithms and datasets. Key observations:
- Substantial performance gains for unstable learners (e.g., decision trees).
- Marginal or no improvement for stable learners.
- Works well for both **classification** and **regression** tasks.
- Improvement is robust to noise in the data.

---

## Practical Considerations

- **Parallelizable**: Each model is trained independently, making Bagging ideal for distributed systems.
- **No hyperparameter tuning**: Simple to implement and works well out-of-the-box.
- **Foundation for Random Forests**: Random Forests are essentially Bagging with an added feature selection step at each tree node, adding even more diversity to the ensemble.

---

## Broader Impact and Legacy

Bagging was one of the first successful ensemble methods in machine learning and remains a key building block in modern techniques. It directly influenced the development of **Random Forests**, **Bootstrap Confidence Intervals**, and inspired contrastive approaches like **Boosting**.

The principle of reducing model variance via ensembling is now core to both traditional ML and modern deep learning (e.g., model ensembling in Kaggle competitions, dropout as implicit bagging).