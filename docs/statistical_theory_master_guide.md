# 🏛️ Basel-Scorecard-Lakehouse: Complete End-to-End Statistical & Quantitative Risk Theory Master Guide

> **Project:** Enterprise Basel Credit Risk Scorecard & Databricks Lakehouse  
> **Domain:** Quantitative Credit Risk Modeling / Decision Science / Model Risk Governance  
> **Regulatory Standards:** Basel II / Basel III IRB (Internal Ratings-Based) Approach, Federal Reserve SR 11-7, Equal Credit Opportunity Act (ECOA, Regulation B)  
> **Author:** Harsh Yadav  

---

## 📑 Table of Contents

1. [Regulatory Credit Risk Architecture (Basel II / III IRB)](#1-regulatory-credit-risk-architecture-basel-ii--iii-irb)
2. [Regression Foundations, Multicollinearity & Matrix Diagnostics](#2-regression-foundations-multicollinearity--matrix-diagnostics)
3. [Weight of Evidence (WoE) & Information Value (IV) Mathematics](#3-weight-of-evidence-woe--information-value-iv-mathematics)
4. [Logistic Modeling, Maximum Likelihood & Convex Optimization](#4-logistic-modeling-maximum-likelihood--convex-optimization)
5. [Points to Double the Odds (PDO) Scorecard Scaling Equation](#5-points-to-double-the-odds-pdo-scorecard-scaling-equation)
6. [Regulatory Validation Metrics & Curve Discrimination Theory](#6-regulatory-validation-metrics--curve-discrimination-theory)
7. [Probability Calibration & Basel Capital Adequacy (Brier Score)](#7-probability-calibration--basel-capital-adequacy-brier-score)
8. [Asymmetric Financial Loss Optimization & Optimal Cutoff ($\tau^*$)](#8-asymmetric-financial-loss-optimization--optimal-cutoff-tau)
9. [Production Drift & Population Stability Index (PSI)](#9-production-drift--population-stability-index-psi)
10. [Model Explainability & Fair Lending Adverse Action (SR 11-7 / ECOA)](#10-model-explainability--fair-lending-adverse-action-sr-11-7--ecoa)

---

## 1. Regulatory Credit Risk Architecture (Basel II / III IRB)

### 1.1 The Foundation: Expected Loss (EL) vs. Unexpected Loss (UL)

In banking risk management, credit losses are decomposed into two distinct probability distributions:

```
  Probability
  Density f(L)
      ▲
      │       EXPECTED LOSS (EL)           UNEXPECTED LOSS (UL)
      │      (Covered by Provisions /     (Covered by Regulatory Equity Capital /
      │       Product Pricing)             Tier 1 Capital Reserves)
      │    ┌──────────────────────┐    ┌───────────────────────────────────┐
      │    │                      │    │                                   │
      │    │                      │    │                                   │
      │   ████                                                             │
      │  ██████                                                            │
      │ ████████                                                           │
      │███████████                                                         │
      │█████████████                                                       │
      │███████████████                                                     │
      │█████████████████                                                   │
      │███████████████████                                                 │
      │█████████████████████                                               │
      │███████████████████████                                             │
      │█████████████████████████                                           │
      │███████████████████████████                                         │
      └───────┴───────────────────────┴───────────────────────────────┴────┴────────► Portfolio Loss ($)
              0                      μ = EL                          99.9% VaR
                                      │                                │
                                      └──────────── UL ────────────────┘
```

1. **Expected Loss ($\text{EL}$):** The anticipated mean annual loss inherent in a loan portfolio under normal business conditions. $\text{EL}$ is treated as a cost of doing business and is absorbed via **loan loss provisions** and credit risk margins built into loan interest rates:
   $$\mathbf{\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}}$$
   * $\mathbf{\text{PD}}$ (**Probability of Default**): The statistical likelihood ($[0, 1]$) that a borrower fails to meet contractual obligations within a 1-year horizon (or lifetime). **This is what our Basel Scorecard models.**
   * $\mathbf{\text{LGD}}$ (**Loss Given Default**): The percentage of exposure lost if default occurs after accounting for recoveries, collateral liquidation, and legal collection costs ($1 - \text{Recovery Rate}$).
   * $\mathbf{\text{EAD}}$ (**Exposure at Default**): The gross dollar exposure outstanding when default takes place (drawn balance plus a Credit Conversion Factor $\text{CCF}$ on undrawn commitments).

2. **Unexpected Loss ($\text{UL}$):** The severe statistical volatility of portfolio losses exceeding the expected mean, driven by economic downturns or systemic shocks. Under Basel II/III, banks must hold **Risk-Weighted Capital (Tier 1 Equity)** up to a **$99.9\%$ Value-at-Risk (VaR)** level:
   $$\mathbf{\text{UL} = \text{VaR}_{0.999} - \text{EL}}$$

> [!IMPORTANT]
> Because multi-billion-dollar minimum equity capital reserves depend directly on the predicted $\text{PD}$, credit scorecards must not only accurately **rank-order** applicants (discrimination), but their output probabilities must also be **strictly calibrated** to actual default rates across all economic vintages.

---

## 2. Regression Foundations, Multicollinearity & Matrix Diagnostics

### 2.1 The OLS Formulation & The 5 Gauss-Markov Classical Assumptions

When fitting linear representations $\hat{y} = X\beta$, Ordinary Least Squares minimizes the Residual Sum of Squares:
$$\mathcal{L}(\beta) = \|y - X\beta\|_2^2 = (y - X\beta)^T(y - X\beta) = y^Ty - 2\beta^TX^Ty + \beta^TX^TX\beta$$

Setting the gradient $\nabla_\beta \mathcal{L} = -2X^Ty + 2X^TX\beta = 0$ yields the **Normal Equation**:
$$\mathbf{\hat{\beta} = (X^TX)^{-1}X^Ty}$$

Under the **Gauss-Markov Theorem**, $\hat{\beta}$ is the **BLUE** (Best Linear Unbiased Estimator) if five conditions hold:
1. **Linearity in parameters:** $y = X\beta + \epsilon$.
2. **Strict Exogeneity:** $\mathbb{E}[\epsilon \mid X] = 0 \implies \text{Cov}(X_j, \epsilon) = 0$ (no omitted variable bias or endogeneity).
3. **No Perfect Multicollinearity:** $\text{Rank}(X) = d \implies \det(X^TX) \ne 0$ (invertibility).
4. **Homoskedasticity:** $\text{Var}(\epsilon_i \mid X) = \sigma^2$ (constant error variance across all values of $X$).
5. **No Serial Autocorrelation:** $\text{Cov}(\epsilon_i, \epsilon_j \mid X) = 0 \quad \forall i \ne j$.

---

### 2.2 The Geometry of Multicollinearity: Eigenvalue Collapse & Exploding Variance

The parameter covariance matrix for any linear or generalized linear model is proportional to the inverse of the feature Gram matrix:
$$\text{Cov}(\hat{\beta}) = \sigma^2 (X^TX)^{-1}$$

Performing spectral eigenvalue decomposition on the symmetric matrix $X^TX$:
$$X^TX = Q \Lambda Q^T = \sum_{j=1}^d \lambda_j q_j q_j^T \implies (X^TX)^{-1} = Q \Lambda^{-1} Q^T = \sum_{j=1}^d \frac{1}{\lambda_j} q_j q_j^T$$

The variance of parameter estimate $\hat{\beta}_j$ is given by:
$$\mathbf{\text{Var}(\hat{\beta}_j) = \sigma^2 \left[ (X^TX)^{-1} \right]_{jj} = \sigma^2 \sum_{k=1}^d \frac{q_{jk}^2}{\lambda_k}}$$

```
                       THE EIGENVALUE COLLAPSE PHENOMENON
                       
    Orthogonal Independent Features                     Near-Collinear Features (e.g., loan_amnt vs installment)
    (Well-Conditioned Hyper-Bowl)                      (Knife-Edge Ravine / Singular Valley)
    λ_1 = 5.0,  λ_2 = 5.0                              λ_1 = 9.9999,  λ_min = 0.0001
    
    Inverse Eigenvalues:                               Inverse Eigenvalues:
    1 / λ_1 = 0.20,  1 / λ_2 = 0.20                    1 / λ_min = 1 / 0.0001 = 10,000!
    Variance & Standard Errors are small and stable!   Variance EXPLODES by 10,000x! Standard errors explode!
```

#### Matrix Condition Number ($\kappa$):
$$\mathbf{\kappa(X^TX) = \frac{\lambda_{\max}}{\lambda_{\min}} = \left(\frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}\right)^2}$$
- $\kappa < 100$: Well-conditioned system.
- $100 \le \kappa \le 1,000$: Moderate collinearity.
- $\kappa > 1,000$: **Severe ill-conditioning**. Inversion is numerically unstable; coefficients swing wildly in sign and magnitude upon minor perturbations in the training dataset.

---

### 2.3 Variance Inflation Factor (VIF) Derivation

To measure the degree to which collinearity inflates the variance of coefficient $\hat{\beta}_j$, we run an **Auxiliary Regression** where predictor $X_j$ is regressed onto all other $d-1$ predictors:
$$X_j = \alpha_0 + \sum_{k \ne j} \alpha_k X_k + \nu_j$$

Let $R_j^2$ be the coefficient of determination from this regression. It can be proven analytically that:
$$\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{(n-1) \text{Var}(X_j)} \times \mathbf{\frac{1}{1 - R_j^2}}$$

The second term is the **Variance Inflation Factor ($\text{VIF}_j$)**:
$$\mathbf{\text{VIF}_j = \frac{1}{1 - R_j^2}}$$

```
┌───────────────────────────┬──────────────────────┬───────────────────┬────────────────────────────────────────┐
│ Auxiliary R_j²            │ VIF_j Value          │ Standard Error SE │ Pipeline Action in Basel Scorecard     │
├───────────────────────────┼──────────────────────┼───────────────────┼────────────────────────────────────────┤
│ **0.00** (Orthogonal)     │ **1.0**              │ Baseline ($1.0x$) │ Retain feature                         │
│ **0.75** (Moderate)       │ **4.0**              │ Inflated $2.0x$   │ Retain feature                         │
│ **0.80** (High)           │ **5.0**              │ Inflated $2.24x$  │ Review / combine feature               │
│ **0.90** (Severe)         │ **10.0**             │ Inflated $3.16x$  │ 🚨 Mandatory Drop (`installment`)      │
│ **0.99** (Extreme)        │ **100.0**            │ Inflated $10.0x$  │ Immediate Drop (Near-singular matrix)  │
└───────────────────────────┴──────────────────────┴───────────────────┴────────────────────────────────────────┘
```

> [!NOTE]
> In our project, when testing `loan_amnt` and `installment`, the auxiliary regression yielded $R^2 \approx 0.922$, resulting in $\text{VIF} = 12.87 \ge 10.0$. The pipeline automatically dropped `installment` and retained `loan_amnt` ($\text{VIF} = 4.21$) and `installment_to_inc` ($\text{VIF} = 1.38$), perfectly preserving numerical stability.

---

## 3. Weight of Evidence (WoE) & Information Value (IV) Mathematics

### 3.1 First-Principles Derivation of Weight of Evidence (WoE)

Weight of Evidence originates from information theory, Alan Turing's wartime cryptographic work, and Bayesian probability.

Let $Y \in \{0, 1\}$ be the binary borrower outcome:
- $Y = 0$: **Good Borrower** (Repaid in full)
- $Y = 1$: **Bad Borrower** (Default / Charged Off)

Let $X$ be a predictor partitioned into $k$ discrete intervals or categories: $\{B_1, B_2, \dots, B_k\}$.  
By Bayes' Rule, the posterior odds of an applicant belonging to bin $B_i$ being a Good borrower rather than a Bad borrower is:

$$\frac{P(Y=0 \mid X \in B_i)}{P(Y=1 \mid X \in B_i)} = \frac{\frac{P(X \in B_i \mid Y=0) P(Y=0)}{P(X \in B_i)}}{\frac{P(X \in B_i \mid Y=1) P(Y=1)}{P(X \in B_i)}} = \mathbf{\left[ \frac{P(X \in B_i \mid Y=0)}{P(X \in B_i \mid Y=1)} \right]} \times \left[ \frac{P(Y=0)}{P(Y=1)} \right]$$

Taking the natural logarithm of both sides:
$$\ln\left( \frac{P(Y=0 \mid X \in B_i)}{P(Y=1 \mid X \in B_i)} \right) = \mathbf{\ln\left( \frac{P(X \in B_i \mid Y=0)}{P(X \in B_i \mid Y=1)} \right)} + \ln\left( \frac{P(Y=0)}{P(Y=1)} \right)$$

$$\mathbf{\text{Posterior Log-Odds}_i = \text{WoE}_i + \text{Prior Log-Odds}}$$

Where the sample empirical estimate of $\text{WoE}_i$ is:
$$\mathbf{\text{WoE}_i = \ln\left( \frac{\text{Goods}_i / \text{Total Goods}}{\text{Bads}_i / \text{Total Bads}} \right) = \ln\left( \frac{\% \text{Goods}_i}{\% \text{Bads}_i} \right)}$$

```
                   PROPERTIES & INTUITIONS OF WoE BINS
                   
    WoE > 0: %Goods_i > %Bads_i ──► LOW-RISK SEGMENT (Higher proportion of good borrowers)
    WoE = 0: %Goods_i = %Bads_i ──► NEUTRAL SEGMENT (Matches portfolio baseline odds)
    WoE < 0: %Goods_i < %Bads_i ──► HIGH-RISK SEGMENT (Higher concentration of defaults)
```

#### Why WoE is Regulated in Banking (Basel Compliance):
1. **Linearizes Non-Linear Relationships:** Logistic regression models linear log-odds: $\ln\left(\frac{p}{1-p}\right) = X\beta$. Replacing non-linear raw continuous variables ($X$) with $\text{WoE}(X)$ ensures a mathematically monotonic linear relationship with the logit link function.
2. **Standardized Scale:** All variables are converted into the exact same dimensionless log-odds scale, making coefficient comparisons and economic interpretation transparent.
3. **Monotonicity Enforcement:** Regulatory rules (Fed SR 11-7) mandate that credit risk must move monotonically with risk attributes (e.g., as FICO increases, default rate must strictly decrease). Adjacent bins that violate monotonicity are algorithmically merged.
4. **Outlier and Missing Handling:** Missing values (`NaN`) and extreme outliers ($>99.9\text{th}$ percentile) are isolated into dedicated bins without distorting model coefficients.

---

### 3.2 Information Value (IV) & Symmetric Kullback-Leibler Divergence

To quantify the total separation power of feature $X$, we compute its **Information Value (IV)**.  
Information Value is mathematically equivalent to the **Symmetric Kullback-Leibler Divergence (Jeffreys Divergence)** between the conditional probability distributions $P(X \mid Y=0)$ and $P(X \mid Y=1)$:

$$D_{\text{KL}}(P_{\text{Good}} \parallel P_{\text{Bad}}) = \sum_{i=1}^k P(X \in B_i \mid Y=0) \ln\left( \frac{P(X \in B_i \mid Y=0)}{P(X \in B_i \mid Y=1)} \right) = \sum_{i=1}^k (\% \text{Goods}_i) \cdot \text{WoE}_i$$

$$D_{\text{KL}}(P_{\text{Bad}} \parallel P_{\text{Good}}) = \sum_{i=1}^k P(X \in B_i \mid Y=1) \ln\left( \frac{P(X \in B_i \mid Y=1)}{P(X \in B_i \mid Y=0)} \right) = -\sum_{i=1}^k (\% \text{Bads}_i) \cdot \text{WoE}_i$$

Summing both directional divergences gives the symmetric distance:
$$\mathbf{\text{IV} = D_{\text{KL}}(P_{\text{Good}} \parallel P_{\text{Bad}}) + D_{\text{KL}}(P_{\text{Bad}} \parallel P_{\text{Good}}) = \sum_{i=1}^{k} \left( \% \text{Goods}_i - \% \text{Bads}_i \right) \times \text{WoE}_i}$$

```
┌───────────────────────────┬───────────────────────────────────┬────────────────────────────────────────┐
│ Information Value (IV)    │ Predictive Strength               │ Pipeline Regulatory Action             │
├───────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ **< 0.02**                │ Unpredictive                      │ **Mandatory Drop**                     │
│ **0.02 – 0.10**           │ Weak Predictor                    │ Retain with economic rationale         │
│ **0.10 – 0.30**           │ **Medium / Strong (PRIME)**       │ **Core Scorecard Feature**             │
│ **0.30 – 0.50**           │ Very Strong                       │ Excellent Scorecard Predictor          │
│ **> 0.50**                │ Suspiciously High                 │ 🚨 **Investigate for Data Leakage**    │
└───────────────────────────┴───────────────────────────────────┴────────────────────────────────────────┘
```

In our LendingClub real run:
* `fico_mid`: $\text{IV} = \mathbf{0.1071}$ (**Prime Basel Predictor**)
* `installment_to_inc`: $\text{IV} = \mathbf{0.0768}$
* `dti`: $\text{IV} = \mathbf{0.0700}$
* `delinq_2yrs`: $\text{IV} = 0.0016$ (Dropped due to insufficient separation power)

---

## 4. Logistic Modeling, Maximum Likelihood & Convex Optimization

### 4.1 The Logit Link Function & Odds Ratios ($e^{\beta}$)

Let $p_i = P(y_i = 1 \mid x_i)$ be the Probability of Default. The logistic model maps linear combinations of features to $[0, 1]$ via the **Sigmoid Activation**:

$$p_i = \sigma(x_i^T \beta) = \frac{1}{1 + e^{-x_i^T \beta}} = \frac{e^{x_i^T \beta}}{1 + e^{x_i^T \beta}}$$

Taking the logit transformation:
$$\mathbf{\ln\left( \frac{p_i}{1 - p_i} \right) = \text{Logit}(p_i) = \beta_0 + \sum_{j=1}^d \beta_j \cdot \text{WoE}_{ij}}$$

#### Economic Meaning of Odds Ratio ($e^{\beta_j}$):
For a unit increase in feature $X_j$:
$$\text{OR}_j = \frac{\text{Odds}(X_j + 1)}{\text{Odds}(X_j)} = e^{\beta_j}$$
* If $\beta_j < 0 \implies \text{OR}_j < 1$: Higher feature values reduce default risk (e.g., FICO WoE).
* If $\beta_j > 0 \implies \text{OR}_j > 1$: Higher feature values increase default risk (e.g., DTI).

---

### 4.2 Maximum Likelihood Estimation (MLE) & Binary Cross-Entropy

For $n$ independent Bernoulli trials $y_i \in \{0, 1\}$, the likelihood function is:
$$L(\beta) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}$$

Taking the natural log gives the Log-Likelihood $\ell(\beta)$:
$$\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

The objective function to minimize is the **Binary Cross-Entropy (BCE) Loss**:
$$\mathbf{\mathcal{L}(\beta) = -\frac{1}{n}\ell(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \ln(\sigma(x_i^T\beta)) + (1 - y_i)\ln(1 - \sigma(x_i^T\beta)) \right]}$$

---

### 4.3 First-Order Gradient & Second-Order Hessian Derivation

Using the sigmoid derivative identity:
$$\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1 - \sigma(z)) = p(1 - p)$$

Differentiating $\mathcal{L}(\beta)$ with respect to $\beta_j$:
$$\frac{\partial \mathcal{L}}{\partial \beta_j} = -\frac{1}{n} \sum_{i=1}^n \left[ \frac{y_i}{p_i} \frac{\partial p_i}{\partial \beta_j} - \frac{1 - y_i}{1 - p_i} \frac{\partial p_i}{\partial \beta_j} \right] = -\frac{1}{n} \sum_{i=1}^n \left[ \frac{y_i - p_i}{p_i(1 - p_i)} \right] \cdot p_i(1 - p_i) x_{ij}$$

$$\mathbf{\nabla_\beta \mathcal{L} = \frac{1}{n} X^T (p - y)}$$

Differentiating a second time gives the **Hessian Matrix ($H$)**:
$$\mathbf{H = \nabla_\beta^2 \mathcal{L} = \frac{1}{n} X^T W X}$$

Where $W$ is an $n \times n$ diagonal weight matrix with entries:
$$W_{ii} = p_i (1 - p_i)$$

#### Proof of Strict Convexity:
Since $p_i \in (0, 1)$, every diagonal entry $W_{ii} = p_i(1 - p_i) > 0$. Therefore, $W$ is strictly positive definite ($W \succ 0$).  
For any non-zero vector $v \in \mathbb{R}^d$:
$$v^T H v = \frac{1}{n} v^T X^T W X v = \frac{1}{n} (Xv)^T W (Xv) = \frac{1}{n} \sum_{i=1}^n W_{ii} (X_i v)^2 \ge 0$$
When $X$ has full column rank ($\text{Rank}(X) = d$), $Xv \ne 0 \implies v^T H v > 0$.  
The Hessian is **strictly positive definite ($H \succ 0$)** everywhere, proving that **the Binary Cross-Entropy loss surface is strictly convex with a unique global minimum and zero local minima.**

---

### 4.4 Optimization: Newton-Raphson & Iteratively Reweighted Least Squares (IRLS)

Because the Hessian is available in closed form, parameter updates follow Newton-Raphson:
$$\beta^{(t+1)} = \beta^{(t)} - H^{-1} \nabla \mathcal{L} = \beta^{(t)} - (X^T W X)^{-1} X^T (p - y)$$

Rearranging into weighted regression form gives **IRLS**:
$$\mathbf{\beta^{(t+1)} = (X^T W X)^{-1} X^T W z}$$
Where $z = X\beta^{(t)} + W^{-1}(y - p)$ is the working response vector.

---

## 5. Points to Double the Odds (PDO) Scorecard Scaling Equation

### 5.1 The Mathematical Transformation (Log-Odds $\to$ FICO Scale)

Raw model output is the log-odds of default: $\text{Logit} = \ln\left(\frac{p}{1-p}\right)$.  
In retail banking, consumer credit scores (e.g., FICO, VantageScore) are scaled such that:
1. Higher scores mean **lower default risk** (inverting the sign).
2. The score changes linearly with log-odds.
3. Every increase of **$\text{PDO}$ points** (Points to Double the Odds) doubles the odds of being a Good borrower (halves default odds).

The linear transformation is:
$$\mathbf{\text{Score} = \text{Offset} - \text{Factor} \times \ln(\text{Odds}_{\text{default}})}$$

Where $\text{Odds}_{\text{default}} = \frac{p}{1-p}$.

---

### 5.2 Analytical Derivation of Factor and Offset

Let the scorecard satisfy two calibration conditions:
1. At **Base Odds** ($\text{Odds}_0 = \frac{1}{\text{BaseOdds}_{\text{good}}}$, e.g. $50:1$ Good-to-Bad $\implies \text{Odds}_0 = \frac{1}{50}$), the score equals the **Base Score** ($S_0$, e.g. $600$):
   $$S_0 = \text{Offset} - \text{Factor} \times \ln(\text{Odds}_0)$$
2. Doubling the odds of a Good borrower (which halves default odds to $\frac{1}{2}\text{Odds}_0$) increases the score by $\text{PDO}$ points:
   $$S_0 + \text{PDO} = \text{Offset} - \text{Factor} \times \ln\left( \frac{1}{2} \text{Odds}_0 \right)$$

Subtracting Equation 1 from Equation 2:
$$\text{PDO} = -\text{Factor} \times \left[ \ln\left(\frac{1}{2}\right) + \ln(\text{Odds}_0) - \ln(\text{Odds}_0) \right] = -\text{Factor} \times (-\ln(2)) = \text{Factor} \times \ln(2)$$

$$\mathbf{\text{Factor} = \frac{\text{PDO}}{\ln(2)}}$$

Substituting $\text{Factor}$ back into Equation 1 yields the Offset:
$$\mathbf{\text{Offset} = S_0 + \text{Factor} \times \ln(\text{Odds}_0) = S_0 - \text{Factor} \times \ln(\text{BaseOdds}_{\text{good}})}$$

---

### 5.3 Concrete Numerical Calibration in our Project

* Base Score $S_0 = 600$ points
* Base Good-to-Bad Odds $= 50:1 \implies \ln(50) \approx 3.9120$
* $\text{PDO} = 20$ points

$$\text{Factor} = \frac{20}{\ln(2)} = \frac{20}{0.693147} \approx \mathbf{28.8539}$$
$$\text{Offset} = 600 - 28.8539 \times \ln(50) = 600 - 28.8539 \times 3.9120 = 600 - 112.876 \approx \mathbf{487.12}$$

$$\mathbf{\text{Score} = 487.12 - 28.8539 \times \left( \beta_0 + \sum_{j=1}^d \beta_j \cdot \text{WoE}_j \right)}$$

---

### 5.4 Additive Scorecard Points Allocation

In commercial scorecard engines, the total score is computed by looking up integer points for each attribute bin and adding them:

$$\text{Score} = \sum_{j=1}^d \text{Points}_{ij}$$

Where the points for bin $i$ of feature $j$ are allocated by distributing the intercept across all $d$ features:
$$\mathbf{\text{Points}_{ij} = \left( \frac{\text{Offset} - \text{Factor} \cdot \beta_0}{d} \right) - \left( \text{Factor} \cdot \beta_j \cdot \text{WoE}_{ij} \right)}$$

This is precisely the additive lookup table exported in [outputs/scorecard_points_table.csv](file:///home/harsh/MLOPs/Basel_Scorecard_Lakehouse/outputs/scorecard_points_table.csv).

---

## 6. Regulatory Validation Metrics & Curve Discrimination Theory

### 6.1 The Kolmogorov-Smirnov (KS) Decile Separation Statistic

The KS statistic is the **primary regulatory validation benchmark** mandated by Basel supervisory authorities to assess a scorecard's ability to separate good borrowers from bad borrowers.

Let $F_{\text{good}}(s)$ and $F_{\text{bad}}(s)$ be the empirical cumulative distribution functions of credit scores for non-defaulters and defaulters:
$$F_{\text{good}}(s) = P(\text{Score} \le s \mid Y=0), \qquad F_{\text{bad}}(s) = P(\text{Score} \le s \mid Y=1)$$

The Kolmogorov-Smirnov statistic is the supremum of the absolute vertical difference:
$$\mathbf{\text{KS} = \max_{s} |F_{\text{bad}}(s) - F_{\text{good}}(s)| \times 100\%}$$

```
                                  KS DECILE SEPARATION MECHANICS
                                  
  Cumulative %
     100% ┼───────────────────────────────────────────────────────────────────────────●
          │                                                       ●───────● Cum % Bads
      80% ┼───────────────────────────────────────────────●───────
          │                                       ●───────
      60% ┼───────────────────────────────●───────
          │                       ●───────        ▲
      40% ┼───────────────●───────                │  MAX KS SEPARATION (Decile 4 = 21.65%)
          │       ●───────                        ▼
      20% ┼───────                                █───────● Cum % Goods
          │                                       │
       0% ┼───────────────────────────────────────┴───────────────────────────────────
          Decile 1     Decile 2     Decile 3   Decile 4   ...   Decile 9    Decile 10
          (Lowest FICO / Highest Risk)                          (Highest FICO / Lowest Risk)
```

```
┌───────────────────────────┬───────────────────────────────────┬────────────────────────────────────────┐
│ KS Value Range            │ Regulatory Classification         │ Supervisory Action                     │
├───────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ **< 20%**                 │ Weak Discrimination               │ **Model Rejected**                     │
│ **20% – 35%**             │ **Acceptable / Solid Production** │ **Approved for Retail IRB Portfolios** │
│ **35% – 55%**             │ **Ideal Basel Scorecard**         │ **Industry Gold Standard**             │
│ **> 60%**                 │ Suspiciously High                 │ 🚨 **Audit for Target Leakage**        │
└───────────────────────────┴───────────────────────────────────┴────────────────────────────────────────┘
```

In our model execution on real 2018 holdout loans:
$$\mathbf{\text{KS}_{\text{OOT}} = 22.86\% \implies \text{PASSES BASEL IRB SUPERVISORY GATE}}$$

---

### 6.2 The Probabilistic Proof of ROC-AUC (Wilcoxon-Mann-Whitney U Statistic)

The Receiver Operating Characteristic (ROC) curve plots True Positive Rate ($\text{TPR} = \text{Recall}$) against False Positive Rate ($\text{FPR} = 1 - \text{Specificity}$) across all decision thresholds $\tau \in [0, 1]$.

#### The Fundamental Probabilistic Theorem:
The Area Under the ROC Curve ($\text{AUC}$) equals the exact probability that a randomly selected Bad borrower ($Y=1$) receives a higher predicted default probability $\hat{p}$ than a randomly selected Good borrower ($Y=0$):
$$\mathbf{\text{ROC-AUC} = P(\hat{p}_{\text{Bad}} > \hat{p}_{\text{Good}})}$$

#### Mathematical Proof via Double Integration:
Let $f_1(s)$ be the probability density function of scores for Bads ($Y=1$), and $F_0(s) = \int_{-\infty}^s f_0(t) dt$ be the cumulative distribution function for Goods ($Y=0$).  
At any cutoff $s$, $\text{TPR}(s) = 1 - F_1(s)$ and $\text{FPR}(s) = 1 - F_0(s)$.  
The Area Under the Curve is:
$$\text{AUC} = \int_{0}^{1} \text{TPR} \, d(\text{FPR}) = \int_{-\infty}^{\infty} (1 - F_1(s)) \left( -\frac{d F_0(s)}{ds} \right) (-ds) = \int_{-\infty}^{\infty} (1 - F_1(s)) f_0(s) \, ds$$
Integrating by parts:
$$\text{AUC} = \int_{-\infty}^{\infty} F_0(s) f_1(s) \, ds = \mathbb{E}_{S_{\text{Bad}}} [P(S_{\text{Good}} < S_{\text{Bad}} \mid S_{\text{Bad}})] = \mathbf{P(S_{\text{Bad}} > S_{\text{Good}})}$$
This confirms that ROC-AUC is purely a measure of **rank concordance**.

---

### 6.3 The Gini Coefficient in Banking

In retail credit risk reporting under Basel III, banks report the **Gini Coefficient** (also known as the Sommer's D or Accuracy Ratio $AR$):

$$\mathbf{\text{Gini} = 2 \times \text{ROC-AUC} - 1}$$

* $\text{ROC-AUC} = 0.50 \implies \text{Gini} = 2(0.50) - 1 = \mathbf{0.00}$ (Random baseline).
* $\text{ROC-AUC} = 1.00 \implies \text{Gini} = 2(1.00) - 1 = \mathbf{1.00}$ (Perfect discrimination).

In our project:
$$\mathbf{\text{Gini}_{\text{OOT}} = 2(0.6631) - 1 = 0.3262}$$

---

## 7. Probability Calibration & Basel Capital Adequacy (Brier Score)

### 7.1 Why Discrimination Alone is Insufficient (The Capital Inadequacy Trap)

A model can have an exceptional $\text{ROC-AUC} = 0.85$ while producing catastrophically uncalibrated default probabilities:

$$\hat{p}_{\text{model}} = 0.01 \quad \text{when true } P(\text{Default}) = 0.20$$

Because Basel regulatory capital is calculated as $\mathbf{\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}}$, underestimating $\text{PD}$ by $10\times$ causes a bank to allocate only $\$10\text{M}$ in equity reserves instead of $\$100\text{M}$. When an economic downturn strikes, the bank becomes insolvent due to under-capitalization.

---

### 7.2 The Brier Score Formulation & Decomposition

The **Brier Score** measures mean squared probability calibration:
$$\mathbf{\text{Brier} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2}$$

Murphy (1973) proved that the Brier Score decomposes into three orthogonal statistical components:
$$\mathbf{\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}}$$

1. **Reliability (Calibration):** $\sum_{k=1}^K w_k (\bar{p}_k - \bar{y}_k)^2$. Measures how close the predicted probability $\bar{p}_k$ in risk decile $k$ is to the observed empirical default rate $\bar{y}_k$. **Must be near 0.**
2. **Resolution (Discrimination):** $\sum_{k=1}^K w_k (\bar{y}_k - \bar{y})^2$. Measures how much each decile's default rate differs from the portfolio average. **Higher is better.**
3. **Uncertainty:** $\bar{y}(1 - \bar{y})$. The intrinsic Bernoulli variance of the portfolio target.

In our retrained model:
$$\mathbf{\text{Brier} = 0.2138 \implies \text{Calibrated for Regulatory Reserve Feeding}}$$

---

## 8. Asymmetric Financial Loss Optimization & Optimal Cutoff ($\tau^*$)

### 8.1 The Fallacy of the Default Threshold $\tau = 0.50$

Standard machine learning libraries default to classifying applicants with $\hat{p} \ge 0.50$ as Defaults ($y=1$).  
In consumer lending, this is financially disastrous because **the cost of errors is heavily asymmetric**:

```
                              THE BANKING COST MATRIX
                              
                          ┌───────────────────────────┬───────────────────────────┐
                          │ Actual Good (y = 0)       │ Actual Bad (y = 1)        │
┌─────────────────────────┼───────────────────────────┼───────────────────────────┤
│ Predict Good (p < τ)    │ TRUE ACCEPT (TN)          │ FALSE ACCEPT (FN)         │
│ (Approve Loan)          │ Profit: +$300             │ Loss: -$10,000            │
│                         │ (Earn Net Interest Margin)│ (Principal Write-off)     │
├─────────────────────────┼───────────────────────────┼───────────────────────────┤
│ Predict Bad (p ≥ τ)     │ FALSE REJECT (FP)         │ TRUE REJECT (TP)          │
│ (Decline Loan)          │ Opportunity Cost: -$300   │ Avoided Loss: $0          │
│                         │ (Lost Interest Margin)    │ (No exposure taken)       │
└─────────────────────────┴───────────────────────────┴───────────────────────────┘
```

* $\mathbf{C_{\text{FP}} = \$300}$: False Positive Cost (declining a good borrower forfeits interest margin).
* $\mathbf{C_{\text{FN}} = \$10,000}$: False Negative Cost (approving a bad borrower forfeits unpaid principal).

---

### 8.2 Mathematical Derivation of Optimal Threshold ($\tau^*$)

For a new loan applicant with predicted default probability $p$:
* If approved ($\hat{y} = 0$), the expected financial cost is:
  $$\mathbb{E}[\text{Cost} \mid \text{Approve}] = p \cdot C_{\text{FN}} + (1 - p) \cdot 0 = p \cdot C_{\text{FN}}$$
* If declined ($\hat{y} = 1$), the expected financial cost is:
  $$\mathbb{E}[\text{Cost} \mid \text{Decline}] = (1 - p) \cdot C_{\text{FP}} + p \cdot 0 = (1 - p) \cdot C_{\text{FP}}$$

The rational bank approves the applicant if and only if:
$$\mathbb{E}[\text{Cost} \mid \text{Approve}] < \mathbb{E}[\text{Cost} \mid \text{Decline}]$$
$$p \cdot C_{\text{FN}} < (1 - p) \cdot C_{\text{FP}}$$
$$p \cdot C_{\text{FN}} < C_{\text{FP}} - p \cdot C_{\text{FP}}$$
$$p (C_{\text{FP}} + C_{\text{FN}}) < C_{\text{FP}}$$

$$\mathbf{\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}}$$

Substituting commercial banking figures:
$$\mathbf{\tau^* = \frac{300}{300 + 10,000} = \frac{300}{10,300} \approx 0.0291 \quad (\mathbf{2.91\%})}$$

> [!IMPORTANT]
> **Operational Rule:** Any applicant whose predicted probability of default exceeds **$2.91\%$** must be declined. Using the naive $\tau = 0.50$ threshold would approve borrowers with a $40\%$ chance of defaulting, leading to massive financial losses.

---

## 9. Production Drift & Population Stability Index (PSI)

### 9.1 Mathematical Formulation of PSI

The **Population Stability Index (PSI)** quantifies whether the score distribution of a live production inference batch has drifted away from the baseline validation cohort.

Let:
* $E_i = \% \text{Expected}_i$: Percentage of applicants in score decile $i$ during baseline training.
* $A_i = \% \text{Actual}_i$: Percentage of applicants in score decile $i$ during live production inference.

$$\mathbf{\text{PSI} = \sum_{i=1}^{k} \left( A_i - E_i \right) \times \ln\left( \frac{A_i}{E_i} \right)}$$

Just like Information Value, PSI is the **Symmetric Kullback-Leibler Divergence** between the baseline and live production distributions:
$$\text{PSI} = D_{\text{KL}}(A \parallel E) + D_{\text{KL}}(E \parallel A)$$

---

### 9.2 Regulatory PSI Threshold Action Matrix

```
┌───────────────────────────┬───────────────────────────────┬────────────────────────────────────────────┐
│ PSI Value                 │ Population Shift Level        │ Automated Production Action                │
├───────────────────────────┼───────────────────────────────┼────────────────────────────────────────────┤
│ **PSI < 0.10**            │ 🟢 **Minimal / Stable**       │ **No action required**; continue inference │
│ **0.10 ≤ PSI < 0.25**     │ 🟡 **Moderate Shift**         │ Issue warning; increase monitoring cadence │
│ **PSI ≥ 0.25**            │ 🔴 **Significant Drift**      │ 🚨 **Mandatory Automated Model Retraining**│
└───────────────────────────┴───────────────────────────────┴────────────────────────────────────────────┘
```

In our 3-Epoch production run:
* **Epoch 2 (2016 Inference):** $\text{PSI} = \mathbf{0.0075} \implies$ **STABLE**
* **Epoch 3 (2017 Live Batch):** $\text{PSI} = \mathbf{0.0411} \implies$ **STABLE**

---

## 10. Model Explainability & Fair Lending Adverse Action (SR 11-7 / ECOA)

### 10.1 Regulatory Compliance Mandate (ECOA Regulation B)

Under the **Equal Credit Opportunity Act (ECOA, 15 U.S.C. 1691 et seq.)** and Federal Reserve **Model Risk Management Guidance (SR 11-7)**, banks cannot deploy "black-box" credit decision engines.  
Whenever an applicant is declined credit or offered unfavorable terms, the creditor **must provide a written Statement of Adverse Action** identifying the **Top 4 principal reasons** that adversely influenced the score.

---

### 10.2 TreeSHAP & LinearExplainer Attributions

For an individual applicant with feature vector $x^*$, SHAP computes the Shapley value $\phi_j$ from cooperative game theory:

$$\phi_j(v) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ v(S \cup \{j\}) - v(S) \right]$$

For our scorecard's logit model $f(x) = \beta_0 + \sum_{j} \beta_j \text{WoE}_j$:
$$f(x^*) - \mathbb{E}[f(X)] = \sum_{j=1}^d \phi_j(x^*)$$

Where the local attribution simplifies to:
$$\mathbf{\phi_j(x^*) = \beta_j \cdot (\text{WoE}_{j}(x^*) - \mathbb{E}[\text{WoE}_j])}$$

```
                               SHAP WATERFALL FOR LOAN #128582025
                               
  Feature (Raw Attribute)          Log-Odds Impact (Pushing to Default)     Assigned Reason Code
  ─────────────────────────────────────────────────────────────────────────────────────────────
  1. dti (High Debt-to-Income)     ████████████████ +0.302                  RC01: Excessive DTI
  2. home_ownership (Rent/Other)   ██████████ +0.197                        RC99: Home Ownership Tier
  3. annual_inc (Low Income)       █████ +0.093                             RC06: Insufficient Income
  4. fico_mid (Subprime Score)     ██ +0.046                                RC03: Low Bureau FICO Score
  ─────────────────────────────────────────────────────────────────────────────────────────────
  Total Default Probability: 52.74% ──► FINAL DECISION: DECLINED (FICO 484)
```

The system automatically extracts the top four positive $\phi_j$ drivers and exports the formal JSON adverse action notice: [outputs/sample_adverse_action_notice.json](file:///home/harsh/MLOPs/Basel_Scorecard_Lakehouse/outputs/sample_adverse_action_notice.json).

---

## 🏁 Architectural Summary Diagram

```
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                THE STATISTICAL SCORECARD PIPELINE                                 │
 └─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 1. MULTICOLLINEARITY (VIF < 10.0)                                                                 │
 │    Auxiliary R_j² regression drops collinear raw features (installment) to prevent H ill-conditioning │
 └─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 2. MONOTONIC WoE & INFORMATION VALUE (IV >= 0.05)                                                 │
 │    WoE_i = ln(%G_i / %B_i); Enforce monotonicity; Rank via Jeffreys Symmetric KL-Divergence       │
 └─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 3. BALANCED LOGISTIC REGRESSION (MLE & BCE LOSS)                                                  │
 │    Strictly convex loss: H = X^T W X ≻ 0; Global convergence via Newton-Raphson / IRLS            │
 └─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 4. PDO SCORECARD SCALING (FICO 300 to 850)                                                        │
 │    Score = 487.12 - 28.854 * Logit (Base 600 @ 50:1 Odds, PDO = 20 points)                        │
 └─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 5. REGULATORY VALIDATION & CALIBRATION                                                            │
 │    KS Decile Separation (22.86%) · Gini (0.3262) · Brier Score (0.2138) · Optimal Cutoff τ* (2.91%) │
 └─────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                   │
                                                   ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ 6. DRIFT RETRAINING & FAIR LENDING EXPLAINABILITY                                                 │
 │    Population Stability Index (PSI < 0.10) · TreeSHAP Top-4 Adverse Action Reason Codes (SR 11-7) │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Appendix: Key Banking & Credit Risk Glossary

### What is a Mortgage?
A **mortgage** is a specialized type of secured loan used to purchase or refinance real estate (such as a house, land, or commercial building), where **the physical property itself serves as collateral** for the debt.

* **The Arrangement:** A lender (bank or financial institution) provides capital upfront. The borrower contributes a cash down payment (typically 5% to 20%) and repays the remaining principal plus interest over an agreed term (commonly 15, 20, or 30 years) through monthly installments comprising principal, interest, taxes, and insurance (PITI).
* **The Legal Lien & Foreclosure:** While the borrower holds legal title and occupies the home, the lender maintains a legal claim (**lien**) on the property deed. If the borrower defaults (fails to make scheduled amortizing payments), the lender possesses the statutory right to seize and auction the property via **foreclosure** to recover the unpaid principal.
* **Credit Risk Significance in Basel IRB Modeling:**
  * **Lower Loss Given Default ($\text{LGD}$):** Because mortgages are collateralized by physical real estate, the bank's net loss upon default is substantially lower than on unsecured credit cards or personal loans. Repossessing and auctioning the property typically recovers 75% to 90% of the loan value ($\text{LGD} \approx 10\%\text{--}25\%$, compared to $60\%\text{--}85\%$ for unsecured loans).
  * **Role in Our Scorecard:** In the LendingClub feature space, variables like `home_ownership` (MORTGAGE vs. RENT vs. OWN) and `mort_acc` (number of mortgage accounts) serve as strong indicators of borrower residential stability, credit maturity, and asset backing.
