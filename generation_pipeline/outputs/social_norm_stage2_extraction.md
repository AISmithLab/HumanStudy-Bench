# Stage 2: Study & Data Extraction Review

## Study 1: Pluralistic Ignorance regarding Alcohol Habits

### Phenomenon
Pluralistic Ignorance

### Sub-Studies/Scenarios

#### study_1_comfort_estimation (questionnaire)
- **Content Preview**: Participants rated their own comfort and estimated the average student's comfort with campus alcohol habits, including an estimate of the interquartile range of the student body.
- **Participants N**: 132
- **Human Data**: {
  "item_level_results": [
    {
      "measure": "Self-comfort (Women)",
      "mean": 4.68,
      "sd": 2.69
    },
    {
      "measure": "Average student comfort (Women's estimate)",
      "mean": 7.07,
      "sd": 1.68
    },
    {
      "measure": "Self-comfort (Men)",
      "mean": 6.03,
      "sd": 2.76
    },
    {
      "measure": "Average student comfort (Men's estimate)",
      "mean": 7.0,
      "sd": 1.57
    },
    {
      "measure": "Self-comfort (Total)",
      "mean": 5.33,
      "sd": 2.73
    },
    {
      "measure": "Average student comfort (Total estimate)",
      "mean": 7.04,
      "sd": 1.63
    }
  ],
  "statistical_results": [
    {
      "test_name": "2 (sex) X 2 (target) analysis of variance (ANOVA)",
      "statistic": "F(1, 130) = 55.52",
      "p_value": "p < .0001",
      "claim": "Highly significant main effect of target: Respondents were much less comfortable than they believed the average student to be.",
      "location": "Page 245, Results and Discussion"
    },
    {
      "test_name": "Sex X Target interaction ANOVA",
      "statistic": "F(1, 130) = 9.96",
      "p_value": "p < .005",
      "claim": "The gap between ratings of own and others' comfort was substantially larger for women than for men.",
      "location": "Page 245, Results and Discussion"
    },
    {
      "test_name": "Statistical comparison of variances (F-test)",
      "statistic": "F(131, 131) = 2.99",
      "p_value": "p < .0001",
      "claim": "Significant difference in variances: students' estimates of the average student's comfort were much less variable than their own comfort ratings.",
      "location": "Page 245, Results and Discussion"
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

## Study 2: Pluralistic Ignorance regarding Friends and Order Effects

### Phenomenon
Pluralistic Ignorance

### Sub-Studies/Scenarios

#### study_2_order_and_friend_comparison (questionnaire)
- **Content Preview**: Participants rated their own comfort, their friends' comfort, and the average student's comfort. The order of questions (self first vs. average student first) was manipulated.
- **Participants N**: 242
- **Human Data**: {
  "item_level_results": [
    {
      "condition": "Self-question first (Total)",
      "self_mean": 5.91,
      "friend_mean": 6.61,
      "average_student_mean": 7.01
    },
    {
      "condition": "Other-question first (Total)",
      "self_mean": 5.41,
      "friend_mean": 6.49,
      "average_student_mean": 7.3
    }
  ],
  "statistical_results": [
    {
      "test_name": "2 (sex) X 3 (target) X 2 (question order) ANOVA",
      "statistic": "F(2, 476) = 54.52",
      "p_value": "p < .0001",
      "claim": "Highly significant main effect of target: ratings of own comfort were significantly lower than ratings of friends' comfort or the average student's comfort.",
      "location": "Page 246, Results and Discussion"
    },
    {
      "test_name": "Target X Order interaction ANOVA",
      "statistic": "F(2, 476) = 3.45",
      "p_value": "p < .05",
      "claim": "Differences between targets were greater when the average student question came first, but remained significant in both orders.",
      "location": "Page 246, Results and Discussion"
    },
    {
      "test_name": "Variance comparison (Self vs Average)",
      "statistic": "F(241, 241) = 3.14",
      "p_value": "p < .0001",
      "claim": "Ratings of the average student were significantly less variable than ratings of the self.",
      "location": "Page 246, Results and Discussion"
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

## Study 4: Pluralistic Ignorance and Campus Alienation regarding Keg Ban

### Phenomenon
Pluralistic Ignorance and Institutional Alienation

### Sub-Studies/Scenarios

#### study_4_keg_ban_alienation (task)
- **Content Preview**: Participants rated their own attitude toward a campus keg ban, estimated the average student's attitude, and indicated willingness to take social action and their connection to the university.
- **Participants N**: 94
- **Human Data**: {
  "item_level_results": [
    {
      "group": "Others-more-negative (Women)",
      "keg_ban_attitude_mean": 4.7,
      "signatures_mean": 6.05,
      "hours_mean": 0.4,
      "reunions_mean": 33.88,
      "donations_mean": 4.65
    },
    {
      "group": "Others-the-same (Women)",
      "keg_ban_attitude_mean": 0.64,
      "signatures_mean": 49.09,
      "hours_mean": 2.55,
      "reunions_mean": 57.27,
      "donations_mean": 6.27
    }
  ],
  "statistical_results": [
    {
      "test_name": "ANCOVA on willingness to collect signatures (controlling for attitudes)",
      "statistic": "F(1, 90) = 18.94",
      "p_value": "p < .0001",
      "claim": "Significant effect of comparison group: those who felt deviant were less willing to collect signatures.",
      "location": "Page 251, Results"
    },
    {
      "test_name": "ANCOVA on willingness to work hours (controlling for attitudes)",
      "statistic": "F(1, 90) = 10.99",
      "p_value": "p < .005",
      "claim": "Significant effect of comparison group: those who felt deviant were less willing to spend time discussing protest.",
      "location": "Page 251, Results"
    },
    {
      "test_name": "ANCOVA on percentage of reunions expected (controlling for attitudes)",
      "statistic": "F(1, 89) = 8.10",
      "p_value": "p < .01",
      "claim": "Significant effect of comparison group: those who felt deviant expected to attend fewer reunions.",
      "location": "Page 251, Results"
    },
    {
      "test_name": "ANOVA on personal attitude (Others-more-negative vs. Others-the-same)",
      "statistic": "F(1, 90) = 48.26",
      "p_value": "p < .0001",
      "claim": "Subjects in the others-more-negative group expressed more favorable attitudes toward the ban than did subjects in the others-the-same group.",
      "location": "Page 251, Results"
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
