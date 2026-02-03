# Stage 2: Study & Data Extraction Review

## Experiment 1: Main Study: Anchoring in Quantitative Estimation

### Phenomenon
Anchoring Effect

### Sub-Studies/Scenarios

#### exp_1_calibration (calibration_group)
- **Content Preview**: A calibration group provided estimates and confidence ratings for 15 uncertain quantities to establish baseline distributions and anchors (15th and 85th percentiles).
- **Participants N**: 53
- **Human Data**: {}
- **Statistical Tests**: 0 test(s)

#### Checklist:
- [ ] Content correctly extracted
- [ ] Participant N correct
- [ ] Human data correctly extracted
- [ ] Statistical tests correctly identified

#### Comments:
[填写]

#### exp_1_anchored_estimation (experimental_group)
- **Content Preview**: Participants performed two tasks for each of 15 quantities: 1) Judged if the quantity was higher or lower than a specified anchor. 2) Estimated the quantity and rated their confidence on a 10-point scale....
- **Participants N**: 103
- **Human Data**: {
  "item_level_results": [
    {
      "question_id": 1,
      "low_anchor_median": 300,
      "high_anchor_median": 1500,
      "anchoring_index": 0.62
    },
    {
      "question_id": 2,
      "low_anchor_median": 8000,
      "high_anchor_median": 42550,
      "anchoring_index": 0.79
    },
    {
      "question_id": 3,
      "low_anchor_median": 100,
      "high_anchor_median": 500,
      "anchoring_index": 0.42
    },
    {
      "question_id": 4,
      "low_anchor_median": 2600,
      "high_anchor_median": 4000,
      "anchoring_index": 0.31
    },
    {
      "question_id": 5,
      "low_anchor_median": 100,
      "high_anchor_median": 400,
      "anchoring_index": 0.62
    },
    {
      "question_id": 6,
      "low_anchor_median": 26,
      "high_anchor_median": 100,
      "anchoring_index": 0.65
    },
    {
      "question_id": 7,
      "low_anchor_median": 50,
      "high_anchor_median": 95,
      "anchoring_index": 0.43
    },
    {
      "question_id": 8,
      "low_anchor_median": 0.6,
      "high_anchor_median": 5.05,
      "anchoring_index": 0.93
    },
    {
      "question_id": 9,
      "low_anchor_median": 1870,
      "high_anchor_median": 1900,
      "anchoring_index": 0.43
    },
    {
      "question_id": 10,
      "low_anchor_median": 1000,
      "high_anchor_median": 40000,
      "anchoring_index": 0.78
    },
    {
      "question_id": 11,
      "low_anchor_median": 10,
      "high_anchor_median": 20,
      "anchoring_index": 0.43
    },
    {
      "question_id": 12,
      "low_anchor_median": 40,
      "high_anchor_median": 60,
      "anchoring_index": 0.33
    },
    {
      "question_id": 13,
      "low_anchor_median": 20,
      "high_anchor_median": 40,
      "anchoring_index": 0.27
    },
    {
      "question_id": 14,
      "low_anchor_median": 30,
      "high_anchor_median": 50,
      "anchoring_index": 0.25
    },
    {
      "question_id": 15,
      "low_anchor_median": 16,
      "high_anchor_median": 16,
      "anchoring_index": 0.0
    }
  ],
  "statistical_results": [
    {
      "test_name": "Mean Anchoring Index (AI)",
      "statistic": "mean AI = .49",
      "p_value": null,
      "claim": "The subject moved almost halfway toward the anchor from the estimate they would have made without it.",
      "location": "Page 1163, Results and Discussion"
    },
    {
      "test_name": "Point-Biserial Correlation",
      "statistic": "mean r = .42",
      "p_value": null,
      "claim": "Measure of the size of the effect: correlation between subjects' estimates and the anchor seen.",
      "location": "Page 1163, Results and Discussion"
    },
    {
      "test_name": "Student's t-test (Asymmetry of Anchoring)",
      "statistic": "t(102) = 7.99",
      "p_value": "p < .01",
      "claim": "The effect of high anchors (median transformed score = 76) was significantly larger than for low anchors (median = 36).",
      "location": "Page 1163, Effects of High and Low Anchors"
    },
    {
      "test_name": "Student's t-test (Extreme Estimates)",
      "statistic": "t(102) = 6.12",
      "p_value": "p < .001",
      "claim": "High anchors yielded 27% extreme estimates (beyond anchor) compared to 14% for low anchors.",
      "location": "Page 1164, Results and Discussion"
    },
    {
      "test_name": "Pearson Correlation (AI vs. Confidence)",
      "statistic": "r = -.68",
      "p_value": "p < .05",
      "claim": "Correlation between the AI and the mean confidence in the calibration group across 15 problems.",
      "location": "Page 1165, Anchoring and Confidence"
    },
    {
      "test_name": "Student's t-test (High Anchor vs. Confidence)",
      "statistic": "t(14) = 2.37",
      "p_value": "p < .05",
      "claim": "Negative correlation (mean r = -.14) between transformed estimates and confidence when anchor is high.",
      "location": "Page 1165, Anchoring and Confidence"
    },
    {
      "test_name": "Student's t-test (Low Anchor vs. Confidence)",
      "statistic": "t(14) = 4.80",
      "p_value": "p < .001",
      "claim": "Positive correlation (mean r = .27) between transformed estimates and confidence when anchor is low.",
      "location": "Page 1165, Anchoring and Confidence"
    },
    {
      "test_name": "Student's t-test (Confidence Ratings Comparison)",
      "statistic": "t(154) = 3.53",
      "p_value": "p < .001",
      "claim": "Estimates were made with greater confidence in anchored groups (3.85) than in calibration group (2.99).",
      "location": "Page 1165, Anchoring and Confidence"
    }
  ]
}
- **Statistical Tests**: 0 test(s)

#### Checklist:
- [ ] Content correctly extracted
- [ ] Participant N correct
- [ ] Human data correctly extracted
- [ ] Statistical tests correctly identified

#### Comments:
[填写]

### Overall Participant Profile
- **Total N**: N/A
- **Population**: N/A
- **Recruitment Source**: N/A
- **Demographics**: {}

#### Checklist:
- [ ] Total N correctly extracted
- [ ] Demographics correctly extracted
- [ ] All available information captured

#### Comments:
[填写]

---

## Experiment 2: Replication with Discredited Anchor

### Phenomenon
Anchoring Effect Persistence

### Sub-Studies/Scenarios

#### exp_2_discredited_anchor (experimental)
- **Content Preview**: A close replication of Exp 1 using the same problems but with an added manipulation intended to discredit the informational value of the anchor.
- **Participants N**: N/A
- **Human Data**: {
  "statistical_results": [
    {
      "test_name": "Percentage of judgments",
      "statistic": "High judged low: 28%; Low judged high: 15%",
      "p_value": null,
      "claim": "The manipulation of credibility did not reduce the anchoring effect.",
      "location": "Page 1164, Results and Discussion"
    }
  ]
}
- **Statistical Tests**: 0 test(s)

#### Checklist:
- [ ] Content correctly extracted
- [ ] Participant N correct
- [ ] Human data correctly extracted
- [ ] Statistical tests correctly identified

#### Comments:
[填写]

### Overall Participant Profile
- **Total N**: N/A
- **Population**: N/A
- **Recruitment Source**: N/A
- **Demographics**: {}

#### Checklist:
- [ ] Total N correctly extracted
- [ ] Demographics correctly extracted
- [ ] All available information captured

#### Comments:
[填写]

---

## Experiment 3: Anchoring in Estimation vs. Willingness to Pay (WTP)

### Phenomenon
Anchoring in Referendum/WTP

### Sub-Studies/Scenarios

#### exp_3_wtp_estimation (experimental)
- **Content Preview**: Compared anchoring in 3 estimation problems and 2 WTP questions for public goods. Subjects made dichotomous judgments (referendum for WTP) before numerical estimates.
- **Participants N**: N/A
- **Human Data**: {
  "statistical_results": [
    {
      "test_name": "Percentage of anchored estimates",
      "statistic": "High anchors: 24% equaled or exceeded highest anchor (vs 4.2% calibration); Low anchor: 15% judged high (vs 4.6% calibration).",
      "p_value": null,
      "claim": "High anchors yielded a large proportion of answers even higher than the anchor.",
      "location": "Page 1164, Results and Discussion"
    }
  ]
}
- **Statistical Tests**: 0 test(s)

#### Checklist:
- [ ] Content correctly extracted
- [ ] Participant N correct
- [ ] Human data correctly extracted
- [ ] Statistical tests correctly identified

#### Comments:
[填写]

### Overall Participant Profile
- **Total N**: N/A
- **Population**: N/A
- **Recruitment Source**: N/A
- **Demographics**: {}

#### Checklist:
- [ ] Total N correctly extracted
- [ ] Demographics correctly extracted
- [ ] All available information captured

#### Comments:
[填写]

---

## Review Status
- **Reviewed By**: [填写]
- **Review Status**: [pending/approved/needs_refinement]
- **Review Comments**: [填写]
- **Action**: [continue_to_final/refine_stage2/back_to_stage1]
