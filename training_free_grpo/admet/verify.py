# training_free_grpo/admet/verify.py
import re
from typing import Any, Dict, List, Tuple, Sequence
from collections.abc import Sequence as ABCSequence
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

    # ========= NEW HARD + PARTIAL REWARD =========
    # 三档 reward，让 GRPO 有对比信号
    if error < 0.4:
        reward = 1.0       # 高质量、近似正确
        correct = True
    elif error < 1.5:
        reward = 0.5       # 部分正确
        correct = False    # 不作为“正确”样本
    else:
        reward = 0.0       # 完全错误
        correct = False

    print("DEBUG reward/error:", reward, error)

    return {
        "reward": reward,
        "correct": correct,
        "y_pred": y_pred,
        "y_true": y_true,
        "error": error,
    }



def verify_func(
    samples: Any,
    responses: Any,
) -> Tuple[List[float], Dict[str, float]]:
    """
    通用的 verify 函数：
    - 支持两种调用方式：
        1) verify_func(sample_dict, response_str)
        2) verify_func(list_of_samples, list_of_responses)
    - 内部统一转成 list 后，用 verify_one 逐个计算 reward 和 correct。
    """

    # -------- 1. 统一 samples 为 list[dict] --------
    # 如果传进来是单个 sample（dict），而不是 list，就包一层 list
    if not isinstance(samples, ABCSequence) or isinstance(samples, (str, bytes)):
        samples_list: List[Dict[str, Any]] = [samples]
    else:
        samples_list = list(samples)

    # -------- 2. 统一 responses 为 list[str] / list[float] --------
    if not isinstance(responses, ABCSequence) or isinstance(responses, (str, bytes)):
        responses_list: List[Any] = [responses]
    else:
        responses_list = list(responses)

    print("\n===== DEBUG verify_func =====")
    print("samples type:", type(samples), "-> normalized to list, len:", len(samples_list))
    print("responses type:", type(responses), "-> normalized to list, len:", len(responses_list))
    print("==============================\n")

    # -------- 3. 对齐长度，避免 zip 截断太多或抛错 --------
    if len(samples_list) != len(responses_list):
        raise ValueError(
            f"verify_func: len(samples)={len(samples_list)} "
            f"!= len(responses)={len(responses_list)}"
        )

    rewards: List[float] = []
    num_correct = 0

    # -------- 4. 逐个调用 verify_one --------
    for sample, resp in zip(samples_list, responses_list):
        # 如果 response 是 float，就转成字符串（或你在 verify_one 里怎么处理就按你的逻辑）
        if not isinstance(resp, str):
            resp_for_verify = str(resp)
        else:
            resp_for_verify = resp

        res = verify_one(sample, resp_for_verify)
        rewards.append(res["reward"])
        if res.get("correct", False):
            num_correct += 1

    # -------- 5. 聚合统计信息 --------
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    pass_rate = num_correct / len(rewards) if rewards else 0.0

    stats = {
        "avg_reward": avg_reward,
        # 这里名称沿用你之前的字段，语义就是“这一批里正确比例”
        "Pass@3": pass_rate,
    }

    return rewards, stats
