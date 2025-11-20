# ---- ADMET PROMPTS ----

PROBLEM_WITH_EXPERIENCE_TEMPLATE = """
You are an expert in ADMET and molecular property prediction.

Your task is to predict the Caco-2 cell permeability (log scale) of the
following molecule from its SMILES.

Before solving, carefully study these helpful reasoning experiences:
{experiences}

Now solve this new problem:

[Molecule SMILES]
{problem}

Please provide:
1. Reasoning about ADMET-relevant molecular features (logP, MW, polarity, HBD/HBA).
2. How these features influence Caco-2 permeability.
3. Final numeric prediction (a single float) at the end of your answer.
"""

SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """
You are an expert in ADMET and molecular property prediction.

Summarize the following trajectory of an ADMET agent predicting Caco-2 permeability
from SMILES. Focus on domain-relevant reasoning:
- lipophilicity (logP)
- molecular weight
- H-bond donors/acceptors
- polarity
- structural motifs influencing permeability

Trajectory:
{trajectory}

Grade: {grade}
Ground truth permeability: {answer}

Produce a concise domain-specific summary of what reasoning occurred.
"""


SINGLE_ROLLOUT_SUMMARY_NO_GT_TEMPLATE = """
You are an expert in ADMET and molecular property prediction.

Summarize this trajectory of an agent predicting Caco-2 permeability from SMILES:

Trajectory:
{trajectory}

Give a concise domain-specific summary focusing on molecular properties.
"""


SINGLE_QUERY_CRITIQUE_TEMPLATE = """
You are an expert in ADMET and Caco-2 molecular permeability analysis.

Your job is to improve the reasoning experience-base for predicting Caco-2 permeability.

==================================================
PROBLEM (Molecule):
{problem}

GROUND TRUTH Caco-2 permeability:
{answer}

CURRENT EXPERIENCE LIBRARY:
{experiences}

TRAJECTORIES (each includes reasoning & predicted value):
{trajectories}
==================================================

Your task:
1. Identify **incorrect or misleading reasoning patterns** in the trajectories.
2. Identify **useful reasoning that should be added as new experience**.
3. Reference the CURRENT EXPERIENCE LIBRARY to decide:
   - Which experiences should be **modified** (option = "modify")
   - Which experiences should be **merged** (option = "merge")
   - Which new experiences should be **added** (option = "add")
4. You MUST output **at most {max_operations} operations**.
5. You MUST output valid JSON inside ```json ... ```.
6. If no update is useful, output an empty array: `[]`.

IMPORTANT — JSON FORMAT (STRICT):
Each operation must be one of the following forms:

ADD:
{
  "option": "add",
  "experience": "a short, standalone rule or insight about Caco-2 permeability"
}

MODIFY:
{
  "option": "modify",
  "modified_from": "G2",
  "experience": "the improved rewritten experience"
}

MERGE:
{
  "option": "merge",
  "merged_from": ["G1", "G4"],
  "experience": "the merged, generalized experience"
}

You MUST NOT include any explanation outside JSON.
Return ONLY JSON inside ```json ... ```.

Begin your output now.
"""



SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = """
You are an expert in ADMET and Caco-2 molecular permeability analysis.

Your job is to improve the reasoning experience-base for predicting Caco-2 permeability.

==================================================
PROBLEM (Molecule):
{problem}

CURRENT EXPERIENCE LIBRARY:
{experiences}

TRAJECTORIES (each includes reasoning & predicted value):
{trajectories}
==================================================

Your task:
1. Identify reasoning mistakes or missing insights.
2. Suggest useful new experiences or corrections.
3. You MUST output at most {max_operations} operations.
4. If nothing needs to be changed, output an empty list `[]`.

STRICT JSON FORMAT:

ADD example:
{
  "option": "add",
  "experience": "new insight..."
}

MODIFY example:
{
  "option": "modify",
  "modified_from": "G3",
  "experience": "rewritten experience..."
}

MERGE example:
{
  "option": "merge",
  "merged_from": ["G1", "G2"],
  "experience": "combined experience..."
}

Return ONLY JSON inside ```json ... ```.
"""

