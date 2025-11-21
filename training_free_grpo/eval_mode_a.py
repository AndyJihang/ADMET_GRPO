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
async def run_grpo_single_sample(worker_agent, problem: str, gt: float,
                                 grpo_n=3, n_steps=5):
    """
    对单个样本跑一次 training-free GRPO。
    和你的 train.py 逻辑保持一致：
        - 多轮 rollout
        - verify_func 提取预测值
    """
    # 当前最优答案（初始化随机）
    best_pred = None
    best_reward = -1e9

    experiences = {}    # 不跨 sample
    rollouts = []       # 不跨 sample

    for step in range(n_steps):

        # 用你的 rollout_dataset 生成 N 个 candidates
        from training_free_grpo.main import rollout_dataset

        # 每一轮需要构造数据结构
        batch_data = [{
            "prompt": problem,
            "groundtruth": gt,
        }] * grpo_n

        rollouts, _ = await rollout_dataset(
            worker_agent=worker_agent,
            data=batch_data,
            rollouts=rollouts,
            verify_func=verify_func,   # 解析 LLM 输出，得到 float
            rollout_filename=None,     # 不保存
            rollout_concurrency=grpo_n,
            task_timeout=60,
            temperature=0.7,
            max_tokens=2048,
        )

        # 遍历 candidates
        for r in rollouts[-grpo_n:]:
            pred = r["answer"]
            rew = reward_fn(pred, gt)

            if rew > best_reward:
                best_reward = rew
                best_pred = pred

    return best_pred


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
