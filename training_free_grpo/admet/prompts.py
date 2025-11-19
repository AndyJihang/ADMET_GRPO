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
You are an expert in ADMET and molecular permeability.

Here are several trajectories (model attempts) for the same prediction task:

Problem (molecule):
{problem}

Ground truth Caco-2 permeability:
{answer}

Existing experiences:
{experiences}

Trajectories:
{trajectories}

Your task:
1. Identify reasoning mistakes.
2. Identify useful correct patterns.
3. Propose at most {max_operations} experience updates, in JSON.

Possible operations:
- "add": create a new experience
- "modify": rewrite an existing experience ("modified_from": experience ID)
- "merge": merge multiple experiences into a new one ("merged_from": list of IDs)

Your output MUST be JSON inside ```json ... ```
"""


SINGLE_QUERY_CRITIQUE_NO_GT_TEMPLATE = """
You are an expert in ADMET and molecular permeability.

Here are several trajectories (model attempts) for the same prediction task:

Problem (molecule):
{problem}

Existing experiences:
{experiences}

Trajectories:
{trajectories}

Your task:
1. Identify patterns.
2. Suggest improvements.
3. Output at most {max_operations} operations in JSON.
"""


BATCH_EXPERIENCE_UPDATE_TEMPLATE = """
You are an ADMET expert. Merge, refine, or rewrite experiences to create better
general rules for predicting Caco-2 permeability.

Current experiences:
{experiences}

Operations (proposed updates):
{updates}

Return the revised plan as JSON inside ```json ... ```
"""
