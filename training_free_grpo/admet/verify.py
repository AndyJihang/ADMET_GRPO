# training_free_grpo/admet/verify.py
import re
from typing import Dict, Any, Tuple
import math

FLOAT_RE = re.compile(r"-?\d+\.?\d*")


def parse_float_from_response(response: str) -> float | None:
    """
    从 LLM 的 response 中抽取最后一个浮点数。
    支持形如：
      "-5.0"
      "<answer>\n\\boxed{-5.0}\n</answer>"
      "0.89"
    """
    if response is None:
        return None

    # 去掉 boxed / answer tag
    text = response.strip()
    # 用正则抓所有数
    matches = FLOAT_RE.findall(text)
    if not matches:
        return None

    try:
        return float(matches[-1])
    except Exception:
        return None


def verify_one(sample, response):
    print("\n===== DEBUG verify_one =====")
    print("sample type:", type(sample))
    print("sample content:", sample)
    print("response:", response)
    print("============================\n")
    y_true = float(sample["groundtruth"])
    y_pred = parse_float_from_response(response)

    if y_pred is None:
        return {
            "reward": 0.0,
            "correct": False,
            "y_pred": None,
            "y_true": y_true,
            "error": None,
        }

    error = abs(y_pred - y_true)

    # Soft reward (good for regression)
    reward = 1.0 / (1.0+math.exp(error))
    print(reward,error)

    # You may define "correct" as being within 1.0
    tol = 1.0
    correct = error <= tol

    return {
        "reward": reward,
        "correct": correct,
        "y_pred": y_pred,
        "y_true": y_true,
        "error": error,
    }



def verify_func(samples: list[Dict[str, Any]], responses: list[str]) -> Tuple[list[float], Dict[str, float]]:
    print("\n===== DEBUG verify_func =====")
    print("samples type:", type(samples))
    print("samples content:", samples)
    print("responses type:", type(responses))
    print("responses content:", responses)
    print("==============================\n")
    rewards: list[float] = []
    num_correct = 0

    for sample, resp in zip(samples, responses):
        res = verify_one(sample, resp)
        rewards.append(res["reward"])
        if res["correct"]:
            num_correct += 1

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    pass_rate = num_correct / len(rewards) if rewards else 0.0

    stats = {
        "avg_reward": avg_reward,
        "Pass@3": pass_rate,  # 这里你可以改名 / 重定义，先复用字段名也行
    }

    return rewards, stats
