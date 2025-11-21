import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from training_free_grpo.train_utils import (
    load_dataset,
    PROBLEM_WITH_EXPERIENCE_TEMPLATE
)
from training_free_grpo.agents import simple_llm_call
from training_free_grpo.verify import verify_one


def load_latest_experience(experiment_dir):
    """
    自动找到最新 step_k/experiences.json
    """
    if not os.path.exists(experiment_dir):
        print(f"[WARN] experiment dir not found: {experiment_dir}")
        return []

    steps = [d for d in os.listdir(experiment_dir) if d.startswith("step_")]
    if len(steps) == 0:
        print("[WARN] no experience found")
        return []

    steps_sorted = sorted(steps, key=lambda x: int(x.split("_")[1]))
    latest_step = steps_sorted[-1]
    exp_file = os.path.join(experiment_dir, latest_step, "experiences.json")

    if not os.path.exists(exp_file):
        print(f"[WARN] no experiences.json found in latest step: {latest_step}")
        return []

    with open(exp_file, "r") as f:
        exp = json.load(f)

    print(f"[INFO] Loaded {len(exp)} experiences from: {exp_file}")
    return exp


def build_problem_with_experience(problem, experience):
    """
    拼接 experience + 原 prompt → final LLM input
    """
    exp_text = ""

    for e in experience:
        traj = e.get("trajectory", [])
        if len(traj) == 0:
            continue
        exp_text += f"- Thought: {traj[0].get('content','')}\n"

    filled = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
        experience=exp_text.strip(), problem=problem
    )
    return filled


def evaluate(experiment_name, dataset, dataset_truncate):
    # 1) Load dataset
    data = load_dataset(dataset, truncate=dataset_truncate)
    print(f"[INFO] Loaded {len(data)} samples")

    # 2) Load experience bank
    exp_dir = os.path.join("data", "admet", "train", experiment_name)
    experience = load_latest_experience(exp_dir)

    errors = []

    # 3) Loop through samples
    for i, s in enumerate(tqdm(data, desc="Eval")):
        problem = s["problem"]
        gt = float(s["groundtruth"])

        # Combine experience + problem
        if len(experience) > 0:
            problem_final = build_problem_with_experience(problem, experience)
        else:
            problem_final = problem

        # Query model
        pred_text = simple_llm_call(problem_final)

        # Verify + extract float
        verify_ret = verify_one({
            "problem": problem_final,
            "groundtruth": gt,
            "response": pred_text,
            "trajectories": []
        })

        pred = verify_ret["answer"]

        err = abs(pred - gt)
        errors.append(err)

    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))

    print("\n============================")
    print("  Evaluation Summary")
    print("----------------------------")
    print(f"Samples   : {len(errors)}")
    print(f"MAE       : {mae:.4f}")
    print(f"RMSE      : {rmse:.4f}")
    print("============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="caco2_wang")
    parser.add_argument("--dataset_truncate", type=int, default=100)
    args = parser.parse_args()

    evaluate(args.experiment_name, args.dataset, args.dataset_truncate)
