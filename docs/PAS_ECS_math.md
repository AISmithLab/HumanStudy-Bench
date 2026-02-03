# PAS/ECS Mathematical Foundations

gitThis document provides the mathematical foundation for the **main benchmark metrics** and reference material for legacy/auxiliary formulas.

**Main metrics (report these only):**
- **PAS** — Probability of Agreement State (raw): alignment of human/agent replication decisions.
- **ECS** — Effect Consistency Score: **correlation-based** (Lin's Concordance Correlation Coefficient, CCC, between human and agent effect profiles).

**Legacy / auxiliary (do not report as primary):** PAS normalized, ECS_Strict (Z-diff based), caricature regression, per-test Z-diff, etc.

## Table of Contents

1. [Per-Test Definitions (PAS only)](#per-test-definitions-pas-only)
2. [PAS Aggregation (Test → Finding → Study)](#pas-aggregation)
3. [ECS (Correlation-based, main)](#ecs-correlation-based-main)
4. [Appendix: ECS_Strict (Z-diff, legacy)](#appendix-ecs_strict-z-diff-legacy)
5. [Edge Cases & Numeric Thresholds](#edge-cases--numeric-thresholds)
6. [Effective Sample Size (n_eff) Definitions](#effective-sample-size-n_eff-definitions)
7. [Effect Size & Standard Error Formulas](#effect-size--standard-error-formulas)
8. [Code Mapping](#code-mapping)

---

## Per-Test Definitions (PAS only)

ECS is **not** defined per test in the main metric; it is defined at aggregate level as correlation-based (see [ECS (Correlation-based, main)](#ecs-correlation-based-main)). Per-test effect sizes and Z-diff are used only for legacy ECS_Strict (appendix).

### PAS (Bayesian Alignment Score) - Per Test

For each statistical test, we compute:

**Posterior Probabilities:**
- $\pi_h$: Posterior probability that an effect exists in human data (0 to 1)
- $\pi_a$: Posterior probability that an effect exists in agent data (0 to 1)

These are derived from Bayes Factors ($BF_{10}$) computed from test statistics (t, F, r, $\chi^2$, etc.).

**PAS Formula:**
$$PAS = \pi_h \pi_a + (1 - \pi_h)(1 - \pi_a)$$

This measures the probability that human and agent are in the same state (both replicated or both null).

**Range:** $PAS \in [0, 1]$, where:
- $PAS = 1$: Perfect alignment (both agree on effect or both agree on null)
- $PAS = 0$: Complete contradiction (one says effect, other says null)

---

## PAS Aggregation

### PAS (Raw) Aggregation: Test → Finding

**Step 1: Convert PAS to r-scale**
For each test $i$ within a finding:
$$r_i = 2 \cdot BAS_i - 1$$

This transforms $BAS_i \in [0,1]$ to $r_i \in [-1,1]$ (correlation-like scale).

**Step 2: Single Test Case**
If finding has only one test ($K=1$):
$$BAS_f^{raw} = BAS_1$$

No aggregation needed.

**Step 3: Multiple Tests - Fisher-z Pooling**
If finding has $K > 1$ tests:

1. **Fisher-z Transform:**
   $$z_i = \tanh^{-1}(r_i) = \frac{1}{2}\ln\frac{1+r_i}{1-r_i}$$

2. **Weighted Average:**
   $$w_i = n_{eff,i} \quad \text{(effective sample size for test } i\text{)}$$
   $$\bar{z}_f = \frac{\sum_{i=1}^K w_i z_i}{\sum_{i=1}^K w_i}$$

3. **Inverse Transform:**
   $$r_f = \tanh(\bar{z}_f)$$

4. **Convert Back to PAS:**
   $$BAS_f^{raw} = \frac{r_f + 1}{2}$$

**Code:** `src/evaluation/stats_lib.py::aggregate_finding_pas_raw()`

### PAS (Normalized) Aggregation: Test → Finding

**Step 1: Compute Human Ceiling**
For each test $i$:
$$H_i = \pi_{h,i}^2 + (1 - \pi_{h,i})^2$$

This represents the maximum possible PAS when agent perfectly matches human posteriors.

**Step 2: Convert to r-scale**
$$r_i = 2 \cdot BAS_i - 1$$
$$r_{h,i} = 2 \cdot H_i - 1$$

**Step 3: Normalized Ratio**
$$r_i' = \frac{r_i}{r_{h,i}}$$

**Step 4: Single Test Case**
If $K=1$:
$$PAS_f^{norm} = r_1'$$

**Step 5: Multiple Tests - Fisher-z Pooling**
If $K > 1$:

1. **Clamp $r_i'$ to valid range:**
   $$r_i' \leftarrow \max(-1 + \epsilon, \min(1 - \epsilon, r_i'))$$
   where $\epsilon = 10^{-6}$

2. **Fisher-z Transform:**
   $$z_i' = \tanh^{-1}(r_i')$$

3. **Weighted Average:**
   $$\bar{z}_f' = \frac{\sum_{i=1}^K w_i z_i'}{\sum_{i=1}^K w_i}$$

4. **Inverse Transform:**
   $$r_f' = \tanh(\bar{z}_f')$$
   $$PAS_f^{norm} = r_f'$$

**Code:** `src/evaluation/stats_lib.py::aggregate_finding_pas_norm()`

### PAS Aggregation: Finding → Study

**Simple Mean Across Findings:**
$$PAS_{study}^{raw} = \frac{1}{F}\sum_{f=1}^F BAS_f^{raw}$$

$$PAS_{study}^{norm} = \frac{1}{F}\sum_{f=1}^F PAS_f^{norm}$$

where $F$ is the number of findings in the study.

**Note:** We use **simple mean** (not weighted by `finding_weight`) per your interpretation: "randomly pick a human finding, how likely this agent can reproduce the effect."

**Code:** `src/evaluation/stats_lib.py::aggregate_study_pas()`

---

## ECS (Correlation-based, main)

**ECS (Effect Consistency Score)** is the **main** effect-consistency metric. It measures construct validity / profile similarity between human and agent effect profiles using **Lin's Concordance Correlation Coefficient (CCC)** — weighted CCC, not Pearson and not Z-diff.

### Effect Size Standardization

All effect sizes are converted to **Cohen's d-equivalent** for consistency:

- **t-test/f-test/anova:** Already $d$ → $\delta = d$
- **correlation** (stored as Fisher z): $r = \tanh(z)$, then $d = \frac{2r}{\sqrt{1-r^2}}$
- **chi-square** (stored as log OR): $d \approx \log(OR) \cdot \frac{\sqrt{3}}{\pi}$
- **Mann-Whitney** (stored as rank-biserial $r_{rb}$): $d = \frac{2r_{rb}}{\sqrt{1-r_{rb}^2}}$
- **binomial** (stored as proportion $p$): $d \approx 2 \cdot \frac{p - p_0}{\sqrt{p_0(1-p_0)}}$ where $p_0 = 0.5$

**Code:** `src/evaluation/stats_lib.py::effect_to_d_equiv()`

### ECS: Weighted Lin's CCC (Concordance Correlation Coefficient)

**Per-Test Effect Profiles:**
- $\boldsymbol{\delta}_h = (\delta_{h,1}, \delta_{h,2}, ..., \delta_{h,M})$: Human effect sizes (Cohen's d)
- $\boldsymbol{\delta}_a = (\delta_{a,1}, \delta_{a,2}, ..., \delta_{a,M})$: Agent effect sizes (Cohen's d)

**Weights for Fairness:**
Within study $j$, each test gets a two-level weight (per finding, per test) so that each study contributes fairly. See code for exact weighting (finding-level and test-level).

**Weighted Lin's CCC:**
$$\rho_c = \frac{2 \cdot \text{cov}_w(x, y)}{\text{var}_w(x) + \text{var}_w(y) + (\mu_x - \mu_y)^2}$$

where $x_i = \delta_{h,i}$, $y_i = \delta_{a,i}$, and weighted mean, variance, and covariance are:
$$\mu_x = \frac{\sum_i w_i x_i}{\sum_i w_i}, \quad \mu_y = \frac{\sum_i w_i y_i}{\sum_i w_i}$$
$$\text{var}_w(x) = \sum_i w_i (x_i - \mu_x)^2, \quad \text{var}_w(y) = \sum_i w_i (y_i - \mu_y)^2$$
$$\text{cov}_w(x,y) = \sum_i w_i (x_i - \mu_x)(y_i - \mu_y)$$

**Range:** $\rho_c \in [-1, 1]$. CCC measures **agreement** (correlation + accuracy/bias); $1$ = perfect agreement.

**Aggregation Levels:**
- **Overall ECS:** CCC across all tests from all studies (weighted)
- **Domain ECS:** CCC within each domain (Cognition, Strategic, Social)
- **Per-study ECS:** CCC within each study (undefined if study has &lt; 3 tests)

**Code:** `src/evaluation/stats_lib.py::compute_ecs_corr()`, `weighted_ccc()`

---

## Appendix: ECS_Strict (Z-diff, legacy)

The Z-diff based metric is **legacy/auxiliary**. Do not report as primary. Kept for reference and the "Caricature Hypothesis."

### ECS_Strict Aggregation: Test → Finding

**Step 1: Collect Z_diff Values**
For each test $i$ within a finding, extract $Z_{diff,i}$ (skip if NaN/Inf).

**Step 2: Single Test Case**
If $K=1$:
$$Z_f = |Z_{diff,1}|$$
$$ECS_f^{strict} = 2(1 - \Phi(Z_f))$$

**Step 3: Multiple Tests - RMS Pooling**
If $K > 1$:
$$Z_f = \sqrt{\frac{1}{K}\sum_{i=1}^K Z_{diff,i}^2}$$

Then convert to ECS:
$$ECS_f^{strict} = 2(1 - \Phi(Z_f))$$

**Code:** `src/evaluation/stats_lib.py::aggregate_finding_ecs()`

### ECS_Strict Aggregation: Finding → Study

**Simple Mean Across Findings:**
$$ECS_{study}^{strict} = \frac{1}{F}\sum_{f=1}^F ECS_f^{strict}$$

**Code:** `generation_pipeline/pipeline.py` (lines ~1591-1600)

---

## Edge Cases & Numeric Thresholds

### Fisher-z Transform Clamping

**Threshold:** $\epsilon = 10^{-6}$

**Rule:** Before applying $\tanh^{-1}(r)$, clamp $r$ to:
$$r \leftarrow \max(-1 + \epsilon, \min(1 - \epsilon, r))$$

**Rationale:** Prevents infinities when $r = \pm 1$.

**Code:** `src/evaluation/stats_lib.py::fisher_z_transform()` (line 1840)

### Normalization Divide-by-Zero Protection

**Threshold:** $|r_h| < 10^{-8}$

**Rule:** If $|r_h| < 10^{-8}$ (i.e., $H \approx 0.5$), then:
- **Single test:** Return $PAS_f^{norm} = 0.0$
- **Multiple tests:** Skip this test in aggregation

**Rationale:** When human evidence is neutral ($\pi_h \approx 0.5$), normalization is undefined/meaningless.

**Code:** `src/evaluation/stats_lib.py::aggregate_finding_pas_norm()` (lines 2020, 2048)

### Empty/Invalid Data Fallbacks

**Empty test list:**
- PAS (raw): Return $0.5$
- PAS (normalized): Return $0.0$
- ECS: Return $0.0$

**All z-values invalid (NaN/Inf):**
- PAS (raw): Fallback to simple mean of PAS values
- PAS (normalized): Fallback to simple mean of normalized ratios (if any valid)
- ECS: Return $0.0$

**Zero/negative weights:**
- If $\sum w_i \leq 0$: Use unweighted mean of z-values

**Code:** See fallback logic in `aggregate_finding_pas_raw()`, `aggregate_finding_pas_norm()`, `aggregate_finding_ecs()`

### NaN/Inf Handling

**Rule:** Skip any test where:
- $z_i$ is NaN or Inf (after Fisher transform)
- $Z_{diff,i}$ is NaN or Inf
- $z_{agg}$ is NaN or Inf (after aggregation)

**Code:** Check `math.isnan()` and `math.isinf()` before including in aggregation.

---

## Effective Sample Size (n_eff) Definitions

The effective sample size $n_{eff}$ is used as weights in Fisher-z pooling. It is computed per test type as follows:

| Test Type | n_eff Formula | Code Location |
|-----------|---------------|---------------|
| **t-test / F-test / ANOVA** | Independent: $n_1 + n_2$<br>Paired/One-sample: $n$ | `compute_n_eff_for_test()` lines 1886-1895 |
| **Correlation** (Pearson/Spearman) | $n$ | `compute_n_eff_for_test()` lines 1896-1899 |
| **Chi-square** | $\sum$ (all contingency table cells) | `compute_n_eff_for_test()` lines 1900-1906 |
| **Mann-Whitney U** | $n_1 + n_2$ | `compute_n_eff_for_test()` lines 1907-1911 |
| **Binomial / Sign Test** | $n$ | `compute_n_eff_for_test()` lines 1912-1915 |
| **Fallback** | $1$ (unweighted) | `compute_n_eff_for_test()` line 1918 |

**Code:** `src/evaluation/stats_lib.py::compute_n_eff_for_test()`

**Note:** n_eff is also computed and stored during `add_statistical_replication_fields()` (lines ~417-435 in `stats_lib.py`).

---

## Effect Size & Standard Error Formulas

### Cohen's d (t-tests, F-tests)

**Effect Size:**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

**Standard Error:**
- **Independent samples:**
  $$SE_d = \sqrt{\frac{n_1 + n_2}{n_1 n_2} + \frac{d^2}{2(n_1 + n_2)}}$$

- **One-sample / Paired:**
  $$SE_d = \sqrt{\frac{1}{n} + \frac{d^2}{2n}}$$

**Code:** `FrequentistConsistency.cohens_d_se()` (lines 1450-1468)

### Fisher's z (Correlations)

**Effect Size:**
$$z = \frac{1}{2}\ln\frac{1+r}{1-r}$$

**Standard Error:**
$$SE_z = \frac{1}{\sqrt{n-3}}$$

**Code:** `FrequentistConsistency.correlation_se()` (lines 1471-1489)

### Log Odds Ratio (Chi-square / 2×2 tables)

**Effect Size:**
$$\log OR = \ln\frac{ad}{bc}$$

**Haldane-Anscombe Correction:** If any cell $(a, b, c, d) = 0$, add $0.5$ to all cells before computation.

**Standard Error:**
$$SE_{\log OR} = \sqrt{\frac{1}{a} + \frac{1}{b} + \frac{1}{c} + \frac{1}{d}}$$

(After correction if needed)

**Code:** `FrequentistConsistency.log_odds_ratio()` and `log_odds_ratio_se()` (lines 1511-1542)

### Rank-Biserial Correlation (Mann-Whitney U)

**Effect Size:**
$$r_{rb} = 1 - \frac{2U}{n_1 n_2}$$

where $U$ is the Mann-Whitney U statistic.

**Standard Error:**
$$SE_{r_{rb}} = \sqrt{\frac{1}{n_1} + \frac{1}{n_2} + \frac{r_{rb}^2}{2(n_1 + n_2)}}$$

**Code:** `FrequentistConsistency.r_rb_se()` (lines 1759-1775)

### Proportion Difference (Binomial)

**Effect Size:**
$$p = \frac{k}{n}$$

**Standard Error:**
$$SE_p = \sqrt{\frac{p(1-p)}{n}}$$

**Code:** `FrequentistConsistency.calculate_consistency_for_binomial()` (lines 1801-1820)

---

## Code Mapping

### Core Aggregation Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `aggregate_finding_pas_raw()` | `src/evaluation/stats_lib.py:1921` | PAS (raw) aggregation at finding level |
| `aggregate_finding_pas_norm()` | `src/evaluation/stats_lib.py:1982` | PAS (normalized) aggregation at finding level |
| `aggregate_finding_ecs()` | `src/evaluation/stats_lib.py:2096` | ECS_Strict aggregation at finding level (appendix) |
| `aggregate_study_pas()` | `src/evaluation/stats_lib.py:2137` | PAS aggregation at study level (calls finding-level functions) |
| `effect_to_d_equiv()` | `src/evaluation/stats_lib.py:1845` | Convert effect sizes to Cohen's d-equivalent |
| `weighted_ccc()` | `src/evaluation/stats_lib.py:3227` | Weighted Lin's CCC (main ECS) |
| `compute_ecs_corr()` | `src/evaluation/stats_lib.py:3301` | ECS computation (CCC overall/domain/study) |
| `weighted_corr()` | `src/evaluation/stats_lib.py:3109` | Weighted Pearson (retained for figures/appendix) |
| `weighted_linreg()` | `src/evaluation/stats_lib.py:3185` | Weighted least squares regression (caricature, legacy) |
| `fisher_z_transform()` | `src/evaluation/stats_lib.py:1827` | Fisher z-transform helper |
| `fisher_z_inverse()` | `src/evaluation/stats_lib.py:1851` | Inverse Fisher z-transform helper |
| `compute_n_eff_for_test()` | `src/evaluation/stats_lib.py:1866` | Compute effective sample size for weighting |

### Pipeline Integration

| Component | Location | Purpose |
|-----------|----------|---------|
| Study-level PAS computation | `generation_pipeline/pipeline.py:1575-1579` | Override `pas_result['score']` and `['normalized_score']` with Fisher-pooled values |
| Study-level ECS_corr computation | `generation_pipeline/pipeline.py:1581-1600` | Compute ECS_corr (correlation-based) and ECS_Strict (appendix) |
| CSV output | `generation_pipeline/pipeline.py:1697-1741` | Write `PAS_Raw`, `ECS_Test`, `Z_Diff`, `Agent_Effect_d`, `Human_Effect_d` columns |
| Summary JSON | `scripts/generate_results_table.py:843-844` | Include `average_pas_raw`, `average_ecs` (ECS_corr), `ecs_strict_overall` (appendix) |

### Effect Size & SE Computation

| Function | Location | Purpose |
|----------|----------|---------|
| `FrequentistConsistency.cohens_d_se()` | `src/evaluation/stats_lib.py:1450` | SE for Cohen's d |
| `FrequentistConsistency.correlation_se()` | `src/evaluation/stats_lib.py:1471` | SE for Fisher's z (correlation) |
| `FrequentistConsistency.log_odds_ratio_se()` | `src/evaluation/stats_lib.py:1528` | SE for log odds ratio |
| `FrequentistConsistency.r_rb_se()` | `src/evaluation/stats_lib.py:1759` | SE for rank-biserial correlation |
| `FrequentistConsistency.calculate_z_diff()` | `src/evaluation/stats_lib.py:1555` | Compute $Z_{diff}$ and ECS from effect sizes and SEs |

### Field Names in Outputs

**CSV (`detailed_stats.csv`):**
- `PAS_Raw`: Per-test raw PAS (PAS) value
- `ECS_Test`: Per-test ECS value
- `Z_Diff`: Per-test $Z_{diff}$ value
- `Agent_Effect_Size`: Per-test agent effect size
- `Human_Effect_Size`: Per-test human effect size

**JSON (`evaluation_results.json`, `benchmark_summary.json`) — report only main:**
- **Main:** `score` = Study-level PAS (raw), Fisher-pooled.
- **Main:** `average_ecs` = Overall ECS (Lin's CCC).
- **Main:** `ecs_corr_study` = Study-level ECS (Lin's CCC).
- Legacy/auxiliary: `normalized_score` (PAS norm), `ecs_strict_study`, `ecs_strict_overall`, `average_pas` → `average_pas_raw`, `average_consistency_score` → `ecs_strict_study`.

---

## Summary

**Report as main metrics only:**
- **PAS (raw):** Aggregates test-level PAS using Fisher-z pooling within findings, then simple mean across findings.
- **ECS (correlation-based):** Standardizes all effect sizes to Cohen's d-equivalent, then computes **Lin's Concordance Correlation Coefficient (CCC)** between human and agent effect profiles (weighted). CCC measures agreement (correlation + accuracy). This is the real ECS.

**Legacy / auxiliary (do not report as primary):** PAS (normalized), ECS_Strict (Z-diff, RMS-pooled), caricature regression, per-test Z-diff.

All aggregations respect edge cases (clamping, divide-by-zero protection, NaN/Inf handling) with explicit numeric thresholds documented above.

