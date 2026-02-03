# Why ECS (Effect Consistency Score) Doesn't Correlate with PAS/APR

## Summary

**Correlation between PAS and ECS: 0.29** (weak correlation, based on 580 test results from mixed_models)

**Note:** This document uses **PAS (Probability Alignment Score)** terminology. PAS aggregates per-test PAS scores using Fisher-z pooling within findings. See [PAS_ECS_math.md](PAS_ECS_math.md) for complete aggregation formulas.

## What Each Metric Measures

### PAS (Probability Alignment Score)
- **What it measures**: Alignment of **posterior probabilities** (beliefs about hypotheses)
- **Per-test formula**: `PAS = π_h+ * π_a+ + π_h- * π_a- + π_h0 * π_a0`
  - Where π = posterior probability of each hypothesis (H+, H-, H0)
- **Aggregation**: Test-level PAS scores are aggregated using **Fisher-z pooling** within findings, then simple mean across findings
- **Depends on**: 
  - **Bayes Factors** (evidence strength from statistical tests)
  - **Sample sizes** (affects precision of Bayes Factors and n_eff weights)
  - **Direction of effects** (whether agent and human agree on direction)
- **Interpretation**: How well do the agent's **beliefs** (posterior probabilities) align with human beliefs?

### ECS (Effect Consistency Score)
- **What it measures**: Similarity of **effect sizes** (magnitude of effects)
- **Per-test formula**: 
  - `Z_diff = (effect_agent - effect_human) / sqrt(SE_agent² + SE_human²)`
  - `ECS = 2 * (1 - Φ(|Z_diff|))` where Φ is the standard normal CDF
- **Aggregation**: Test-level Z_diff values are aggregated using **RMS pooling** within findings: `Z_f = sqrt(mean(Z_diff²))`, then converted to ECS_f, then simple mean across findings
- **Depends on**:
  - **Cohen's d** (or other effect size metrics like Fisher's z, log OR, rank-biserial correlation)
  - **Standard errors** of effect sizes
  - **Magnitude** of the difference between agent and human effect sizes
- **Interpretation**: How similar are the **effect sizes** (magnitudes) between agent and human?

### APR (Average Pass Rate)
- **What it measures**: Proportion of tests where agent replicates human findings
- **Formula**: `APR = (tests where agent is significant AND direction matches) / (total tests)`
- **Depends on**:
  - **Statistical significance** (p < 0.05)
  - **Direction matching** (same sign of effect)
- **Interpretation**: How often does the agent **replicate** the human finding (significant + correct direction)?

## Why They Don't Correlate

### 1. **Different Quantities**
- **PAS**: Measures **evidence strength** (Bayesian posteriors from Bayes Factors, aggregated via Fisher-z pooling)
- **ECS**: Measures **effect magnitude** (frequentist effect sizes like Cohen's d, aggregated via RMS pooling)
- These are fundamentally different types of information!

### 2. **Sample Size Effects**
**Scenario A**: Large effect size difference, but similar posteriors
- Human: d = 0.8, n = 100 → strong evidence (high π)
- Agent: d = 0.4, n = 400 → also strong evidence (high π) due to larger n
- **Result**: Low ECS (different effect sizes), but High PAS (similar posteriors)

**Scenario B**: Similar effect sizes, but different posteriors
- Human: d = 0.5, n = 200 → moderate evidence (moderate π)
- Agent: d = 0.5, n = 50 → weak evidence (low π) due to smaller n
- **Result**: High ECS (similar effect sizes), but Low PAS (different posteriors)

### 3. **Statistical Power Differences**
- **PAS** is sensitive to **statistical power** (ability to detect effects)
  - High power → strong evidence → high π
  - Low power → weak evidence → low π
- **ECS** is sensitive to **effect size magnitude** regardless of power
  - Same effect size → same ECS (if SEs are similar)
  - Different effect size → different ECS

### 4. **Bayes Factor vs Effect Size**
- **PAS** depends on **Bayes Factors** (BF10), which convert test statistics to evidence
  - BF10 = P(Data | H1) / P(Data | H0)
  - Large BF → high posterior probability
  - Aggregated via Fisher-z pooling (weights by n_eff)
- **ECS** depends on **effect sizes** (Cohen's d, etc.), which measure magnitude
  - d = (mean1 - mean2) / pooled_SD
  - Large d → large effect
  - Aggregated via RMS pooling of Z_diff

### 5. **Direction Handling**
- **PAS** explicitly checks **direction matching** (`direction_match` field)
  - If direction doesn't match, PAS is heavily penalized in the 3-way posterior calculation
  - Direction mismatch → different posterior distributions → low PAS
- **ECS** captures direction **indirectly** through effect size comparison
  - Formula: `Z_diff = (effect_agent - effect_human) / SE_combined`
  - If directions match (both positive or both negative): difference is small if magnitudes are similar → high ECS
  - If directions don't match (opposite signs): difference is large → low ECS
  - **Empirical evidence**: 
    - When direction matches: mean ECS = 0.30
    - When direction mismatches: mean ECS = 0.09 (much lower!)
    - Opposite effect size signs: mean ECS = 0.06 (very low!)
  
  So ECS **does** respond to direction, but through the effect size difference rather than an explicit direction check.

## Example Cases

### Case 1: High PAS, Low ECS
- Human: t = 3.0, n = 100 → d = 0.6, π = 0.95 (strong evidence)
- Agent: t = 2.8, n = 100 → d = 0.56, π = 0.92 (strong evidence)
- **PAS**: High (0.95 * 0.92 = 0.87) - similar posteriors
- **ECS**: Low (0.6 vs 0.56, but SEs make Z_diff large) - different effect sizes

### Case 2: Low PAS, High ECS
- Human: t = 2.0, n = 50 → d = 0.4, π = 0.70 (moderate evidence)
- Agent: t = 2.0, n = 50 → d = 0.4, π = 0.70 (moderate evidence)
- But: **Direction mismatch** (human +, agent -)
- **PAS**: Low (direction mismatch penalizes PAS)
- **ECS**: High (same effect size magnitude)

### Case 3: Both Low
- Human: t = 4.0, n = 200 → d = 0.8, π = 0.98
- Agent: t = 1.0, n = 50 → d = 0.2, π = 0.55
- **PAS**: Low (very different posteriors)
- **ECS**: Low (very different effect sizes)

## Implications

1. **PAS and ECS measure complementary aspects**:
   - PAS: "Do agents believe the same things as humans?" (evidence alignment, aggregated via Fisher-z pooling)
   - ECS: "Do agents produce similar effect sizes as humans?" (magnitude alignment, aggregated via RMS pooling)

2. **Both metrics are important**:
   - High PAS + High ECS = Perfect replication (both evidence and magnitude match)
   - High PAS + Low ECS = Evidence matches but magnitudes differ (might be due to sample size)
   - Low PAS + High ECS = Magnitudes match but evidence differs (might be due to power differences)
   - Low PAS + Low ECS = Poor replication (both evidence and magnitude differ)

3. **For mixed_models (random sampling)**:
   - Low correlation (r = 0.29) suggests that random sampling produces:
     - Sometimes similar effect sizes but different evidence strength
     - Sometimes similar evidence strength but different effect sizes
   - This is expected because random sampling doesn't preserve the relationship between effect sizes and statistical power

## Conclusion

ECS and PAS/APR don't correlate strongly because they measure fundamentally different aspects of replication:
- **ECS**: Effect size magnitude similarity (frequentist), with **indirect** direction sensitivity
  - Captures direction through effect size difference: opposite signs → large difference → low ECS
  - Aggregated via RMS pooling: `Z_f = sqrt(mean(Z_diff²))`
  - Mean ECS when direction matches: 0.30
  - Mean ECS when direction mismatches: 0.09
- **PAS**: Evidence strength alignment (Bayesian), with **explicit** direction checking
  - Directly checks `direction_match` and penalizes mismatches in posterior calculation
  - Aggregated via Fisher-z pooling: `z_i = atanh(2*PAS_i-1)`, weighted by n_eff
- **APR**: Statistical significance replication (frequentist binary)
  - Requires both significance AND direction match to count as "pass"

**Key difference**: ECS responds to direction through the effect size comparison (if signs differ, the difference is large), while PAS explicitly checks direction matching. Both care about direction, but in different ways!

**For complete mathematical details**, including aggregation formulas, edge cases, and code mapping, see [PAS_ECS_math.md](PAS_ECS_math.md).

A complete replication assessment should consider all three metrics together!
