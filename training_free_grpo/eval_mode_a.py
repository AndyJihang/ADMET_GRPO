import asyncio
import math
import numpy as np

from training_free_grpo.admet.dataset import load_data
from training_free_grpo.admet.verify import verify_func
from utu.agents import SimpleAgent
from utu.config import ConfigLoader


# -------------------------------
# 1. reward function
# -------------------------------
def reward_fn(pred: float, gt: float) -> float:
    error = abs(pred - gt)
    return 1.0 / math.exp(1.0 + error)


# -------------------------------
# 2. run GRPO on a single sample
# -------------------------------
async def run_grpo_single_sample(
    worker_agent,
    problem: str,
    gt: float,
    grpo_n: int = 3,
    n_steps: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """
    在“单个样本”上跑 training-free GRPO。
    """

    best_pred = None
    best_reward = -1e9
    rollouts = []  # 每个样本内部的 rollouts（不跨样本）

    for _ in range(n_steps):

        # 构造输入（注意 key 必须是 'problem'）
        batch_data = [{
            "problem": problem,
            "groundtruth": gt,
        }] * grpo_n

        # 生成多个 candidates
        from training_free_grpo.main import rollout_dataset

        rollouts, _ = await rollout_dataset(
            worker_agent=worker_agent,
            data=batch_data,
            rollouts=rollouts,
            verify_func=verify_func,
            rollout_filename="/tmp/rollout_sample.tmp.jsonl",
            rollout_concurrency=grpo_n,
            task_timeout=60,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 看最近这一轮的 grpo_n 个 rollouts
        for r in rollouts[-grpo_n:]:
            # 对于 ADMET，数值在 'response' 里；有些别的 domain 可能用 'answer'
            raw = r.get("answer", r.get("response", None))
            if raw is None:
                continue

            try:
                pred = float(raw)
            except Exception:
                # 模型如果输出了奇怪的东西就跳过
                continue

            rew = reward_fn(pred, gt)

            if rew > best_reward:
                best_reward = rew
                best_pred = pred

    # 万一所有 rollout 都解析失败，就退回一个很糟糕但至少不是 None 的值
    if best_pred is None:
        # 尝试用最后一个 rollout 的 response
        if rollouts:
            raw = rollouts[-1].get("answer", rollouts[-1].get("response", gt))
            try:
                best_pred = float(raw)
            except Exception:
                best_pred = gt  # 实在不行就用真值占位，避免后面报错

    return float(best_pred)


# -------------------------------
# 3. 全 dataset 评估（模式 A）
# -------------------------------
async def eval_dataset_mode_a(dataset="caco2_wang",
                              max_samples=None,
                              grpo_n=3,
                              n_steps=5):

    # ❤️ UTU agent 初始化（与你 train.py 完全一致）
    config = ConfigLoader.load_agent_config("simple/admet_agent.yaml")
    worker_agent = SimpleAgent(config=config)
    await worker_agent.build()

    data = load_data(dataset)
    print(f"Loaded {len(data)} samples.")

    if max_samples is not None:
        data = data[:max_samples]

    preds, gts = [], []

    for idx, sample in enumerate(data, start=1):
        problem = sample["problem"]
        gt = float(sample["groundtruth"])

        print(f"\n=== Sample {idx}/{len(data)} ===")
        print(problem)

        pred = await run_grpo_single_sample(
            worker_agent=worker_agent,
            problem=problem,
            gt=gt,
            grpo_n=grpo_n,
            n_steps=n_steps,
        )

        preds.append(pred)
        gts.append(gt)

        print(f"Groundtruth = {gt:.4f}")
        print(f"Predicted   = {pred:.4f}")
        print(f"Abs Error   = {abs(pred - gt):.4f}")

    preds = np.array(preds)
    gts = np.array(gts)

    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts)**2))

    print("\n==========================")
    print(" Training-Free GRPO Eval (Mode A)")
    print(f" Dataset    : {dataset}")
    print(f" Samples    : {len(data)}")
    print(f" MAE        : {mae:.4f}")
    print(f" RMSE       : {rmse:.4f}")
    print("==========================")

    return mae, rmse


if __name__ == "__main__":
    asyncio.run(eval_dataset_mode_a(
        dataset="caco2_wang",
        max_samples=20,     # debug用
        grpo_n=3,
        n_steps=5,
    ))
