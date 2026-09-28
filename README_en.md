# Exoplanet Classification and Property Prediction

## Project Overview

This project is built on NASA's **Kepler** mission data for exoplanet discovery (`exoplanets_2018.csv`) and has two main parts:

1. **Logistic Regression** to classify a candidate planet as `CONFIRMED` (a real planet) or `FALSE POSITIVE` (a spurious detection).
2. **Linear Regression** to predict a planet's equilibrium temperature (`koi_teq`) from its other orbital and physical properties.

All algorithms (linear regression, logistic regression, gradient descent, regularization) were implemented **from scratch using only NumPy**, with no ready-made libraries such as scikit-learn.

---

## Part 1: Data Cleaning & Preprocessing

### 1. Removing unsuitable columns

- **Columns that cause data leakage** (e.g. `koi_score`, `koi_pdisposition`, `koi_fpflag_*`): these are derived from the same process that determines `koi_disposition`, so keeping them would let the model cheat instead of learning.
- **Uninformative columns** (e.g. identifiers and statistical error margins such as `_err1`, `_err2`): removed to simplify the problem.

### 2. Filtering the data

- `CANDIDATE` rows were removed so the task stays a clear binary classification: `CONFIRMED` vs `FALSE POSITIVE`.
- Rows with missing values were dropped (`dropna`), going from **7197** rows to **6730** rows.

### 3. Class balance

| Class | Share |
|---|---|
| FALSE POSITIVE | 65.1% |
| CONFIRMED | 34.9% |

This distribution is acceptable and not severely imbalanced, so no special handling such as oversampling was needed.

### 4. Handling outliers with a log transform

Some astronomical features (e.g. `koi_period`, `koi_prad`, `koi_depth`, `koi_teq`, `koi_insol`) are highly skewed for genuine physical reasons, not because of data errors. `log1p` was applied to compress the range and improve training stability.

### 5. Encoding and splitting

- A `target` column was added (1 = confirmed, 0 = false positive).
- The data was shuffled (`sample(frac=1)`) and split **80% train / 20% test**.
- All features were **standardized** (subtract the mean, divide by the standard deviation) using statistics from the training set only. The same values were applied to the test set to avoid information leakage.

---

## Part 2: Logistic Regression (Classification)

- **Model:** `z = w·x + b`, followed by `sigmoid(z)`.
- **Cost function:** Binary Cross-Entropy.
- **Training:** manual Gradient Descent (learning rate = 2, with the cost monitored every 100 iterations).
- **Goal:** predict the probability that a candidate planet is real (`CONFIRMED`).

---

## Part 3: Linear Regression (Predicting temperature `koi_teq`)

### Problem 1: Discovering data leakage

In the first experiment, the model reached **R² = 0.997** on the test set, which is unusually high for a simple linear model. Investigation showed that the `koi_insol` column (the amount of stellar radiation reaching the planet) was still among the features.

**Cause:** `koi_insol` and the temperature `koi_teq` are linked by a nearly direct physical relationship (`T ∝ Insolation^0.25`). In practice it is the same information restated, not an independent feature, so the model was "cheating" rather than learning a real relationship.

**Fix:** `koi_insol` was removed from the features, along with the `target` column (which had been left in by mistake from the classification stage).

### A wrong turn: removing `koi_period`

After removing `koi_insol`, we also tried removing `koi_period` (orbital period), assuming it was tied to temperature. Performance collapsed to **R² = 0.46**.

**Reason:** `koi_period` is a fully independent measurement (computed from the timing of the planet's transits across its star) and is not derived from the same measurement as `koi_teq`. The strong relationship between them is real physics (Kepler's laws: a planet closer to its star has a shorter orbit and a higher temperature), not a duplicate of the same information. So it was restored to the features.

### General rule learned

The right question is not *"is the feature correlated with the target?"* (every useful feature will be), but:

> **"Is this feature computed from the same measurement as the target, or from the same underlying raw phenomenon?"**

If yes → **leakage**. If no, and it provides additional independent information → **legitimate feature**.

---

## Bugs fixed during development

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | Cost blew up to NaN | Mixed up `f_wb` (an old global) and `f_wb_r` when computing the gradient | Use the correct variable names tied to the regression |
| 2 | Cost still NaN even with a very small alpha | `mu_r` and `segma_r` were computed from `X_train` (classification data) instead of `x_train_r`, causing column misalignment | Compute `mu_r`/`segma_r` from `x_train_r` itself |
| 3 | Changing `lambda_` (regularization) had no effect | `compute_gradient_r` used `w_r` (a constant global = zeros) instead of the local parameter `w_` | Use the correct `w_` inside the regularization term |
| 4 | `print(w_r)` showed zeros despite training | `gradient_descent_r` returned the values (`return w_, b_`) but the call site didn't capture them | `w_r, b_r = gradient_descent_r(...)` |
| 5 | Test cost was inaccurate | `cost_r` divided by `m` (a constant global = training set size) regardless of the size of the data actually passed in | Compute `m = len(y_train_r)` inside the function on every call |

---

## Final Results (after all fixes)

| Metric | Value |
|---|---|
| Training cost | ≈ 0.0052 |
| Test cost | ≈ 0.0063 |
| R² on test set | ≈ 0.99 |

The small gap between training and test cost shows the model generalizes well with no obvious overfitting. The high R² (now legitimate after removing the leakage) reflects a genuine, strong physical relationship between a planet's orbital properties and its temperature.

---

## Key Lessons Learned

1. An unexpectedly high performance number (like R² = 0.997) is a **red flag**, not a reason to celebrate. Investigate it before accepting it as a final result.
2. **Data leakage** doesn't have to be technical (like using the answer column itself). It can be physical or logical: a feature computed from the same measurement as the target.
3. Updating a variable inside a Python function stays local unless you `return` it and capture it outside. Forgetting this gives the false impression that something "isn't working" even though the logic is correct.
4. Using global variables (like `m`) inside general-purpose functions is risky. Each function should compute what it needs from the parameters passed to it, so it works correctly on any data (train or test) without silent errors.
