# Stage 1: Replicability Filter Review

## Paper Information
- **Title**: Subjective Probability: A Judgment of Representativeness
- **Authors**: Daniel Kahneman, Amos Tversky
- **Abstract**: This paper explores a heuristic—representativeness—according to which the subjective probability of an event, or a sample, is determined by the degree to which it: (i) is similar in essential characteristics to its parent population; and (ii) reflects the salient features of the process by which it is generated. This heuristic is explicated in a series of empirical examples demonstrating predictable and systematic errors in the evaluation of uncertain events. In particular, since sample size does not represent any property of the population, it is expected to have little or no effect on judgment of likelihood. This prediction is confirmed in studies showing that subjective sampling distributions and posterior probability judgments are determined by the most salient characteristic of the sample (e.g., proportion, mean) without regard to the size of the sample. The present heuristic approach is contrasted with the normative (Bayesian) approach to the analysis of the judgment of uncertainty.

## Experiments Overview

### Experiment 1: Similarity of Sample to Population: Birth Sequences
- **Input**: Participants were asked to estimate the number of families (out of 72) with birth orders BGBBBB compared to a standard sequence GBGBBG, and also to compare BBBGGG to GBBGBG.
- **Participants**: 92 high school students in Israel (ages 15-18).
- **Output**: Estimates of frequency and judgments of relative likelihood.
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 2: Majority-Minority Relation: High School Programs
- **Input**: A scenario describing two high school programs with different gender ratios (65% vs 45% boys). Participants guess which program a random class with 55% boys belongs to.
- **Participants**: 89 high school students in Israel.
- **Output**: Categorical choice (Program A or Program B).
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 3: Binomial Outcomes Comparison
- **Input**: Given a binomial process with p = 4/5, participants judge the likelihood of a sample of 10 successes and 0 failures versus 6 successes and 4 failures.
- **Participants**: High school students (part of the general 1500 respondent pool).
- **Output**: Relative likelihood judgments.
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 4: Replicability of Significance (Professional Intuitions)
- **Input**: A statistical problem asking for the probability that a second small group (N=10) will yield a significant result given a first group (N=20) was significant (p < .05).
- **Participants**: Participants of a meeting of the Mathematical Psychology Group and the American Psychological Association (sophisticated psychologists).
- **Output**: Subjective probability estimates (0.0 to 1.0).
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 5: Reflection of Randomness: Marble Distribution
- **Input**: Participants are shown two distributions of 20 marbles among 5 children: one nearly uniform (4,4,4,4,4) and one slightly non-uniform (4,4,5,4,3) and asked which is more probable.
- **Participants**: 52 high school students.
- **Output**: Choice between Distribution I and Distribution II.
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 6: Subjective Sampling Distributions (N-Invariance)
- **Input**: Participants produced subjective sampling distributions for three populations (sexes, heartbeat type, height) across three sample sizes (N = 10, 100, 1000).
- **Participants**: Nine groups of high school students (total ~1500 respondents).
- **Output**: Median probability estimates for various distribution categories.
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 7: Ordinal Judgments of Sample Size (Hospitals, Words, Heights)
- **Input**: Three problems (Hospitals/Babies, Word Length, Median Heights) where participants judge if an extreme outcome is more likely in a large or small sample.
- **Participants**: 97 Stanford undergraduates with no background in statistics.
- **Output**: Choice (Larger, Smaller, or About the same).
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 8: Posterior Probability: Symmetric Binomial Task (Decks of Cards)
- **Input**: 10 symmetric binomial problems varying population proportion, sample ratio, and sample difference. Participants estimate the probability a sample came from a specific deck.
- **Participants**: 10 groups of subjects (average 56 per group).
- **Output**: Median subjective posterior probability estimates.
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

### Experiment 9: Posterior Probability: Height Odds
- **Input**: Participants judge the odds that a sample came from a male population based on (i) a single 5'10" individual or (ii) 6 people with an average height of 5'8".
- **Participants**: 115 students from the University of Michigan with at least one statistics course.
- **Output**: Subjective odds estimates.
- **Replicable**: YES
- **Exclusion Reasons**: None

#### Checklist:
- [ ] No visual input required
- [ ] No time perception/measurement required
- [ ] Participant profile can be constructed
- [ ] Has quantitative/statistical data
- [ ] Can be replicated with LLM agents

#### Comments:
[填写]

---

## Overall Assessment
- **Overall Replicable**: YES
- **Confidence**: 1.00
- **Notes**: The paper is a classic example of cognitive psychology research using vignettes and word problems. Since all stimuli are textual and the required outputs are numerical estimates or categorical choices, it is perfectly suited for replication using LLM agents. The demographic profiles provided are sufficient to construct appropriate system prompts (e.g., high schoolers vs. professional psychologists).

## Review Status
- **Reviewed By**: [填写]
- **Review Status**: [pending/approved/needs_refinement]
- **Review Comments**: [填写]
- **Action**: [continue_to_stage2/refine_stage1/exclude]
