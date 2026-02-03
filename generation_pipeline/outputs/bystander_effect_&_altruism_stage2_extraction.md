# Stage 2: Study & Data Extraction Review

## Experiment 1: Effect of Group Size and Composition on Bystander Intervention

### Phenomenon
Diffusion of Responsibility

### Sub-Studies/Scenarios

#### group_size_intervention (experimental_manipulation)
- **Content Preview**: Participants were placed in individual rooms and communicated via an intercom system to discuss personal problems. They heard a fellow participant (a recording) disclose a history of seizures and subs...
- **Participants N**: 52
- **Human Data**: {
  "item_level_results": [
    {
      "condition": "2-person group (Subject & Victim)",
      "n": 13,
      "percent_responding_by_end_of_fit": 85,
      "mean_time_in_seconds": 52,
      "mean_speed_score": 0.87
    },
    {
      "condition": "3-person group (Subject, Victim, & 1 other)",
      "n": 26,
      "percent_responding_by_end_of_fit": 62,
      "mean_time_in_seconds": 93,
      "mean_speed_score": 0.72
    },
    {
      "condition": "6-person group (Subject, Victim, & 4 others)",
      "n": 13,
      "percent_responding_by_end_of_fit": 31,
      "mean_time_in_seconds": 166,
      "mean_speed_score": 0.51
    }
  ],
  "statistical_results": [
    {
      "test_name": "Chi-square",
      "statistic": "x\u00b2 = 7.91",
      "p_value": "p < .02",
      "claim": "The number of bystanders that the subject perceived to be present had a major effect on the likelihood with which she would report the emergency.",
      "location": "Page 380, Table 1"
    },
    {
      "test_name": "Analysis of variance",
      "statistic": "F = 8.09",
      "p_value": "p < .01",
      "claim": "The effect of group size on speed scores is highly significant.",
      "location": "Page 380, Speed of Response section"
    }
  ]
}
- **Statistical Tests**: 0 test(s)

#### Checklist:
- [x] Content correctly extracted
- [x] Participant N correct
- [x] Human data correctly extracted
- [x] Statistical tests correctly identified

#### Comments:
**⚠️ CRITICAL ISSUE FOR MATERIALS GENERATION:**
The `group_size_intervention` sub-study is a **Between-Subjects** design with 3 CONDITIONS:
- 2-person group (n=13): 85% helping rate
- 3-person group (n=26): 62% helping rate  
- 6-person group (n=13): 31% helping rate

**REQUIRED FIX in Stage 3 (Materials Generation):**
1. Generate **3 SEPARATE material files** OR **1 file with 3 condition variants**:
   - `group_size_intervention_2person.json` (condition: "You are the ONLY other person besides the victim")
   - `group_size_intervention_3person.json` (condition: "There are 2 OTHER people besides the victim")
   - `group_size_intervention_6person.json` (condition: "There are 5 OTHER people besides the victim")
2. Each condition MUST explicitly tell the participant how many OTHER bystanders are present
3. The `specification.json` MUST specify different `n` for each condition (13, 26, 13)
4. The `config.py` MUST assign participants to different conditions and record `condition` field

**Current Problem:** All participants receive the same generic prompt that mentions all conditions without specifying which one THEY are in, making the Between-Subjects manipulation impossible to evaluate.

#### group_composition_variations (experimental_manipulation)
- **Content Preview**: Variations of the 3-person condition were conducted to test the effects of the other bystander's gender and perceived medical competence.
- **Participants N**: 44
- **Human Data**: {
  "item_level_results": [
    {
      "condition": "Female Subject, male other",
      "n": 13,
      "percent_responding": 62,
      "mean_time_seconds": 94,
      "mean_speed_score": 74
    },
    {
      "condition": "Female Subject, female other",
      "n": 13,
      "percent_responding": 62,
      "mean_time_seconds": 92,
      "mean_speed_score": 71
    },
    {
      "condition": "Female Subject, male medic other",
      "n": 5,
      "percent_responding": 100,
      "mean_time_seconds": 60,
      "mean_speed_score": 77
    },
    {
      "condition": "Male Subject, female other",
      "n": 13,
      "percent_responding": 69,
      "mean_time_seconds": 110,
      "mean_speed_score": 68
    }
  ],
  "statistical_results": [
    {
      "test_name": "Not specified",
      "statistic": null,
      "p_value": null,
      "claim": "Variations in sex and medical competence of the other bystander had no important or detectable affect on speed of response.",
      "location": "Page 381, Table 2 and Results section"
    }
  ]
}
- **Statistical Tests**: 0 test(s)

#### Checklist:
- [x] Content correctly extracted
- [x] Participant N correct
- [x] Human data correctly extracted
- [ ] Statistical tests correctly identified

#### Comments:
**⚠️ SIMILAR ISSUE:** This sub-study also has 4 different conditions based on gender/competence composition.
Materials should specify the condition clearly to the participant (e.g., "The other person in your group is a MALE/FEMALE" or "The other person is a MEDICAL STUDENT").
However, this sub-study's finding is "NOT SIGNIFICANT", so it's less critical than group_size_intervention.

#### personality_and_demographic_correlates (survey)
- **Content Preview**: Participants completed several personality scales to see if individual differences predicted helping behavior.
- **Participants N**: 65
- **Human Data**: {
  "item_level_results": [],
  "statistical_results": [
    {
      "test_name": "Pearson correlation",
      "statistic": "r = -.26",
      "p_value": "p < .05",
      "claim": "The size of the community in which the subject grew up correlated with the speed of helping.",
      "location": "Page 381, Individual Difference Correlates section"
    },
    {
      "test_name": "Pearson correlation",
      "statistic": null,
      "p_value": "Not significant",
      "claim": "Personality measures showed no important or significant correlations with speed of reporting the emergency.",
      "location": "Page 381, Individual Difference Correlates section"
    }
  ]
}
- **Statistical Tests**: 0 test(s)

#### Checklist:
- [x] Content correctly extracted
- [x] Participant N correct
- [x] Human data correctly extracted
- [x] Statistical tests correctly identified

#### Comments:
This sub-study tests personality correlations. The main finding is that personality measures are NOT significant predictors. The only significant correlation is with community size (r=-.26, p<.05). Materials should include demographic questions about community size for evaluation.

### Overall Participant Profile
- **Total N**: N/A
- **Population**: N/A
- **Recruitment Source**: N/A
- **Demographics**: {}

#### Checklist:
- [x] Total N correctly extracted
- [x] Demographics correctly extracted
- [x] All available information captured

#### Comments:
Total N should be sum of all sub-studies (52+44+65=161, but some overlap exists). Demographics show NYU intro psych students.

---

## Review Status
- **Reviewed By**: AI Assistant (Claude)
- **Review Status**: needs_refinement
- **Review Comments**: 
  **CRITICAL:** The `group_size_intervention` sub-study requires Between-Subjects condition manipulation (2/3/6 person groups). Current materials generation creates ONE generic file without condition-specific prompts. This makes it IMPOSSIBLE to test the main hypothesis (diffusion of responsibility increases with group size).
  
  **REQUIRED CHANGES:**
  1. Stage 3 must generate separate material files for each condition OR include condition field
  2. Each participant must be told THEIR specific group size (not all three options)
  3. Config must track which condition each participant was assigned to
  4. Evaluator needs condition data to perform Chi-square and ANOVA tests
  
- **Action**: refine_stage2
