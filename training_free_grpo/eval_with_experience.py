import os
import json
import argparse
import asyncio
import re
import numpy as np
from tqdm import tqdm

from utu.agents import SimpleAgent
from utu.config import ConfigLoader

# 和 train.py 保持一致
from training_free_grpo.admet.dataset import load_data
from training_free_grpo.admet.verify import verify_func
from training_free_grpo.admet.prompts import PROBLEM_WITH_EXPERIENCE_TEMPLATE
from training_free_grpo.main import rollout_dataset


def load_latest_experiences(domain: str, experiment_name: str):
    """
    在 data/<domain>/train/<experiment_name>/ 里找到
    最大的 step_k/experiences.json，并加载成 dict。
    """
    base_dir = os.path.join("data", domain, "train", experiment_name)
    if not os.path.exists(base_dir):
        print(f"[WARN] experiment dir not found: {base_dir}")
        return {}

    step_dirs = [
        d for d in os.listdir(base_dir)
        if d.startswith("step_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not step_dirs:
        print(f"[WARN] no step_* dirs in {base_dir}")
        return {}

    step_idx = max(int(d.split("_")[1]) for d in step_dirs)
    latest_step_dir = os.path.join(base_dir, f"step_{step_idx}")
    exp_path = os.path.join(latest_step_dir, "experiences.json")

    if not os.path.exists(exp_path):
        print(f"[WARN] experiences.json not found in {latest_step_dir}")
        return {}

    with open(exp_path, "r", encoding="utf-8") as f:
        experiences = json.load(f)

    print(f"[INFO] Loaded {len(experiences)} experiences from {exp_path}")
    return experiences


def format_experiences_for_prompt(experiences: dict, max_count: int = 20) -> str:
    """
    把 dict 形式的经验格式化成字符串，和 train.py 的用法一致：
    "[key]. value"
    """
    if not experiences:
        return "None"

    lines = []
    for i, (k, v) in enumerate(experiences.items()):
        if i >= max_count:
            break
        lines.append(f"[{k}]. {v}")
    return "\n".join(lines)


def parse_float_from_response(text: str) -> float:
    """
    从 LLM 的输出里提取最后一个浮点数。
    例如：
    "... Final numeric prediction:\n-2.5"
    """
    # 找到所有形如 -2.5, 3, 1e-3 的数字
    matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if not matches:
        raise ValueError(f"Could not find any float in response: {text[:200]}")
    return float(matches[-1])


async def eval_with_experiences(
    experiment_name: str,
    dataset_name: str,
    dataset_truncate: int | None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    domain = "admet"

    # 1. 加载数据
    data = load_data(dataset_name)
    print(f"[INFO] Loaded {len(data)} samples from dataset={dataset_name}")
    if dataset_truncate is not None:
        data = data[:dataset_truncate]
        print(f"[INFO] Truncated to first {len(data)} samples")

    # 2. 加载最终 step 的经验
    experiences = load_latest_experiences(domain, experiment_name)
    formatted_experiences = format_experiences_for_prompt(experiences)

    # 3. 初始化 UTU Agent
    config = ConfigLoader.load_agent_config("simple/admet_agent.yaml")
    config.model.model_settings.temperature = temperature
    worker_agent = SimpleAgent(config=config)
    await worker_agent.build()

    preds = []
    gts = []

    # 4. 逐样本评估
    for idx, sample in enumerate(tqdm(data, desc="Evaluating with experiences")):
        problem_raw = sample["problem"]
        gt = float(sample["groundtruth"])

        # 把经验塞到模板里
        prompt = PROBLEM_WITH_EXPERIENCE_TEMPLATE.format(
            experiences=formatted_experiences,
            problem=problem_raw,
        )

        batch = [{
            "problem": prompt,
            "groundtruth": gt,
        }]

        # 只 rollout 一次
        rollouts, _ = await rollout_dataset(
            worker_agent=worker_agent,
            data=batch,
            rollouts=[],
            verify_func=verify_func,
            rollout_filename="/tmp/admet_eval_with_exp.jsonl",
            rollout_concurrency=1,
            task_timeout=60,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        out = rollouts[-1]
        response_text = out.get("response", "")

        # 直接从文本里解析 float，不用 verify_one
        pred = parse_float_from_response(response_text)

        preds.append(pred)
        gts.append(gt)

        if idx < 3:
            print("\n--- DEBUG SAMPLE", idx, "---")
            print("Groundtruth:", gt)
            print("Predicted  :", pred)
            print("Abs Error  :", abs(pred - gt))

    preds = np.array(preds)
    gts = np.array(gts)

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts) ** 2))

    print("\n==========================")
    print(" Training-Free GRPO Eval")
    print("  (with experiences)")
    print("==========================")
    print(f"Experiment : {experiment_name}")
    print(f"Dataset    : {dataset_name}")
    print(f"Samples    : {len(gts)}")
    print(f"MAE        : {mae:.4f}")
    print(f"RMSE       : {rmse:.4f}")
    print("==========================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="caco2_wang")
    parser.add_argument("--dataset_truncate", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    args = parser.parse_args()

    asyncio.run(
        eval_with_experiences(
            experiment_name=args.experiment_name,
            dataset_name=args.dataset,
            dataset_truncate=args.dataset_truncate,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
