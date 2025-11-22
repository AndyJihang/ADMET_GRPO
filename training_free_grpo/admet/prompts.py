# training_free_grpo/admet/prompts.py
# -*- coding: utf-8 -*-
"""
Prompt templates for ADMET (Caco-2 permeability) in Training-Free GRPO.
包含：
- PROBLEM_WITH_EXPERIENCE_TEMPLATE
- SINGLE_ROLLOUT_SUMMARY_TEMPLATE
- SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE
- SINGLE_QUERY_CRITIQUE_TEMPLATE
- SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE
- BATCH_EXPERIENCE_UPDATE_TEMPLATE
"""

# ==== 1. 问题 + Experience 的主 Prompt =======================================

PROBLEM_WITH_EXPERIENCE_TEMPLATE = """
You are an expert in ADMET and molecular property prediction.

Your task is to predict the Caco-2 cell permeability (log scale, log(cm/s)) of
the following molecule from its SMILES.

IMPORTANT DOMAIN RANGE GUIDANCE:
- Typical experimental Caco-2 permeability values (Wang dataset and similar) 
  usually fall between **-7.5 and -3.0** log(cm/s).
- Values **above -3.0** are biologically uncommon and rarely observed.
- Very low-permeability compounds (high MW, high polarity, peptides, many HB donors)
  often lie in the **-6.5 to -7.5** region.
Use this range to calibrate your numeric prediction.

Before solving, carefully study these helpful reasoning experiences:
{experiences}

Now solve this new problem:

[Molecule SMILES]
{problem}

Please provide:
1. Reasoning about ADMET-relevant molecular features (e.g., logP, molecular weight, polarity, hydrogen bond donors/acceptors).
2. Explain how these features influence Caco-2 permeability.
3. At the very end of your answer, output the final numeric prediction
   as a single float on a separate line (no explanation, no units).
"""

"""

# ==== 2. 单条 rollout 的总结 Prompt ===========================================

SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """
You are an expert in ADMET and molecular property prediction.

Summarize the following trajectory of an ADMET agent predicting Caco-2 permeability
from SMILES. Focus on domain-relevant reasoning such as:
- lipophilicity (logP)
- molecular weight
- hydrogen bond donors/acceptors
- polarity / polar surface area
- charge state
- structural motifs influencing permeability (e.g., aromatic rings, bulky groups, rigid vs flexible backbone)

Trajectory:
{trajectory}

Grade: {grade}
Ground truth Caco-2 permeability (log scale): {answer}

Produce a concise, domain-specific summary of what reasoning occurred.
Do NOT add any new facts that are not implied by the trajectory.
"""

SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE = """
You are an expert in ADMET and molecular property prediction.

Summarize this trajectory of an agent predicting Caco-2 permeability from SMILES.

Trajectory:
{trajectory}

Give a concise domain-specific summary, focusing on:
- which molecular properties are discussed (logP, MW, HBD/HBA, PSA, charge, etc.),
- how these properties were linked to permeability,
- any obvious reasoning mistakes.

Do NOT include any ground-truth information (you do not know it).
"""

# ==== 3. 单个问题层面的 critique Prompt（有 GT） =============================

SINGLE_QUERY_CRITIQUE_TEMPLATE = """
You are an expert in ADMET and Caco-2 molecular permeability analysis.

Your job is to improve the reasoning experience-base for predicting Caco-2 permeability.

==================================================
PROBLEM (Molecule):
{problem}

GROUND TRUTH Caco-2 permeability (log(cm/s)):
{answer}

CURRENT EXPERIENCE LIBRARY (ID -> text):
{experiences}

TRAJECTORIES (each includes reasoning and a predicted value):
{trajectories}
==================================================

Your task:
1. Identify incorrect, incomplete, or misleading reasoning patterns in the trajectories
   (for example, overemphasizing one feature, ignoring polarity, wrong direction of effect, etc.).
2. Identify genuinely useful reasoning patterns that could be turned into short, general experiences.
3. Based on CURRENT EXPERIENCE LIBRARY, decide which experiences should be:
   - modified (option = "modify"),
   - merged (option = "merge"),
   - added as new (option = "add").
4. You MUST output at most {max_operations} operations.
5. If you do not see any useful improvement, output an empty list [].

You MUST output valid JSON inside ```json ... ``` with the following formats ONLY:

ADD:
{{
  "option": "add",
  "experience": "a short, standalone rule or insight about Caco-2 permeability"
}}

MODIFY:
{{
  "option": "modify",
  "modified_from": "G2",
  "experience": "the improved rewritten experience text"
}}

MERGE:
{{
  "option": "merge",
  "merged_from": ["G1", "G4"],
  "experience": "a merged, generalized experience that replaces them"
}}

Constraints:
- Each operation MUST be a JSON object with an "option" field.
- The whole output MUST be a JSON array: [ {{...}}, {{...}} ] or [].
- Do NOT include any natural language explanation outside the JSON.

Return ONLY JSON inside ```json ... ```.
"""

# ==== 4. 单个问题层面的 critique Prompt（无 GT） ============================

SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = """
You are an expert in ADMET and Caco-2 molecular permeability analysis.

Your job is to improve the reasoning experience-base for predicting Caco-2 permeability.

==================================================
PROBLEM (Molecule):
{problem}

CURRENT EXPERIENCE LIBRARY (ID -> text):
{experiences}

TRAJECTORIES (each includes reasoning and a predicted value):
{trajectories}
==================================================

Your task:
1. Identify reasoning mistakes or missing insights across the trajectories.
2. Suggest useful new experiences or corrections that would generally improve future reasoning.
3. You MUST output at most {max_operations} operations.
4. If nothing needs to be changed, output an empty list [].

Valid JSON operation formats:

ADD:
{{
  "option": "add",
  "experience": "new, generalizable insight about Caco-2 permeability..."
}}

MODIFY:
{{
  "option": "modify",
  "modified_from": "G3",
  "experience": "rewritten, improved experience text..."
}}

MERGE:
{{
  "option": "merge",
  "merged_from": ["G1", "G2"],
  "experience": "a combined, more general experience..."
}}

The entire output MUST be a JSON array (e.g., [ {{...}}, {{...}} ]).
Return ONLY JSON inside ```json ... ```.
"""

# ==== 5. 批量 update experiences 的 Prompt ===================================

BATCH_EXPERIENCE_UPDATE_TEMPLATE = """
You are an expert editor of ADMET reasoning experiences for Caco-2 permeability.

You are given:
1. A candidate experience library (ID -> text), which already includes:
   - the original experiences, and
   - newly added candidate experiences.

2. A list of update requests ("updates") that indicate which experience IDs
   are suggested to be modified or merged. Each update has the form:
   - {{"option": "modify", "modified_from": "<ID>", "experience": "<new text>"}} OR
   - {{"option": "merge", "merged_from": ["<ID1>", "<ID2>", ...], "experience": "<merged text>"}}

Your job is to produce a FINAL, clean revision plan that:
- keeps the experience library compact,
- removes redundant or overlapping experiences,
- applies reasonable modifications / merges,
- ensures that each experience is a clear, general rule about Caco-2 permeability.

==================================================
CURRENT CANDIDATE EXPERIENCES (ID -> text):
{experiences}

REQUESTED UPDATES (list of operations):
{updates}
==================================================

OUTPUT REQUIREMENTS:
1. Your output MUST be valid JSON inside ```json ... ```.
2. The JSON MUST be a list (array) of operations, e.g.:
   [
     {{
       "option": "modify",
       "modified_from": "G2",
       "experience": "new text..."
     }},
     {{
       "option": "merge",
       "merged_from": ["G1", "C0"],
       "experience": "merged text..."
     }}
   ]

3. Allowed "option" values: "modify", "merge".
   (You MAY also include new "add" operations if truly necessary.)

4. For "modify":
   - "modified_from" MUST be an existing ID in the candidate experiences.
   - "experience" MUST be the improved text.

5. For "merge":
   - "merged_from" MUST be a non-empty list of existing IDs.
   - "experience" MUST be the new, merged text that replaces them.

6. If, after reviewing, you think NO changes are needed, output an empty list: [].

Do NOT include any explanation outside the JSON.
Return ONLY JSON inside ```json ... ```.
"""
