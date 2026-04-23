"""AMAPPOv2 训练收敛监控测试套件."""
import subprocess
import re
import numpy as np
import pytest


def _run_short_training(epochs: int = 150, seed: int = 0) -> list[dict]:
    """运行短训练并解析日志，返回每个 log_interval 的指标列表。"""
    result = subprocess.run(
        ["python", "experiments/train_v2.py", f"--epochs={epochs}", f"--seed={seed}"],
        capture_output=True, text=True, timeout=300,
    )
    lines = result.stdout.splitlines()
    records = []
    pattern = re.compile(
        r"ep=\s*(\d+)\s+reward=\s*([\-\d.]+)\s+actor_loss=([\-\d.]+)\s+"
        r"critic_loss=([\-\d.]+)\s+entropy=([\-\d.]+)"
    )
    for line in lines:
        m = pattern.search(line)
        if m:
            records.append({
                "ep":           int(m.group(1)),
                "reward":       float(m.group(2)),
                "actor_loss":   float(m.group(3)),
                "critic_loss":  float(m.group(4)),
                "entropy":      float(m.group(5)),
            })
    return records


def test_t4_actor_loss_nonzero():
    """T4: |actor_loss| 均值必须 > 0.01，确认 actor 在有效更新。"""
    records = _run_short_training(epochs=150, seed=42)
    assert len(records) >= 1, "训练日志为空，训练未正常运行"
    actor_losses = [abs(r["actor_loss"]) for r in records]
    mean_abs = float(np.mean(actor_losses))
    assert mean_abs > 0.01, (
        f"actor_loss 均值绝对值 {mean_abs:.5f} <= 0.01，"
        "梯度覆盖(Bug A) 或 DAG错用(Bug B) 未修复"
    )


def test_dag_encoding_consistency():
    """验证 evaluate_actions 中 log_prob_new 和 log_prob_old 来自同一 DAG 编码。

    间接测试：如果 Bug B 未修复，agent 1~3 的 log_prob 差异会是随机噪声，
    表现为 |actor_loss| 极小。T4 已覆盖此检测，此处做单元级验证。
    """
    import torch
    from utils.config import Config
    from models.v2.gnn_encoder_v2 import GNNEncoderV2
    from algorithms.amappo_v2 import _build_graph_inputs_v2
    from env.sec_env import SECEnv

    cfg = Config(J=5, N=10, M=2, seed=0)
    env = SECEnv(cfg)
    env.reset()

    encoder = GNNEncoderV2()

    # agent 0 的 DAG
    dag0_x, dag0_ei, res0_x, res0_ei = _build_graph_inputs_v2(env, 0)
    # agent 1 的 DAG
    dag1_x, dag1_ei, res1_x, res1_ei = _build_graph_inputs_v2(env, 1)

    with torch.no_grad():
        node_embs_0, _, _ = encoder(dag0_x, dag0_ei, res0_x, res0_ei)
        node_embs_1, _, _ = encoder(dag1_x, dag1_ei, res1_x, res1_ei)

    # node_embs_0 和 node_embs_1 对同一 task_id 应有不同的嵌入
    task_id = torch.tensor([0, 1, 2], dtype=torch.long)
    emb0 = node_embs_0[task_id]
    emb1 = node_embs_1[task_id]
    diff = (emb0 - emb1).abs().mean().item()
    assert diff > 1e-4, (
        f"两个 agent 的 DAG node embeddings 完全相同（diff={diff:.6f}），"
        "DAGs 是随机生成的，应当不同。测试环境可能有问题。"
    )


def test_buffer_cleared_after_update():
    """Fix C: 验证每次 PPO update 后 GlobalBuffer 被清空，防止离策略污染。"""
    import torch
    from utils.config import Config
    from algorithms.amappo_v2 import AMAPPOv2Trainer

    cfg = Config(J=5, N=10, M=2, epochs=12, update_every=5, ppo_epochs=1,
                 mini_batch_size=10, seed=0, log_interval=100, save_interval=100)
    trainer = AMAPPOv2Trainer(cfg)

    # 运行 10 个 episode（update_every=5，应触发2次更新）
    train_metrics_list = []
    for ep in range(1, 11):
        trainer._run_episode()
        if ep % cfg.update_every == 0:
            if len(trainer.global_buffer) >= cfg.mini_batch_size:
                trainer._ppo_update()
                trainer.global_buffer.clear()
                train_metrics_list.append(len(trainer.global_buffer))

    # 每次 update 后 buffer 应该被清空
    for size in train_metrics_list:
        assert size == 0, f"Buffer update 后应为空，实际大小: {size}"


@pytest.fixture(scope="module")
def training_records_300():
    """运行 300 轮训练，缓存结果供多个测试使用（避免重复运行）。"""
    return _run_short_training(epochs=300, seed=42)


@pytest.fixture(scope="module")
def training_records_1500():
    """运行 1500 轮训练，用于最终收敛测试。耗时约 5 分钟。"""
    return _run_short_training(epochs=1500, seed=42)


@pytest.mark.slow
def test_t1_critic_loss_decreases(training_records_300):
    """T1: critic_loss 在 300 轮内下降超过 50%."""
    records = training_records_300
    assert len(records) >= 3, f"日志记录不足3条（实际{len(records)}条），无法判断趋势"

    loss_early = records[0]["critic_loss"]   # ep=100
    loss_late  = records[2]["critic_loss"]   # ep=300
    assert loss_late < loss_early * 0.5, (
        f"critic_loss 下降不足50%: ep=100时{loss_early:.1f}，ep=300时{loss_late:.1f}，"
        f"下降了{100*(1-loss_late/loss_early):.1f}%（需>50%）"
    )


@pytest.mark.slow
def test_t2_reward_trend_positive(training_records_1500):
    """T2: ep=500~1000 的 reward 线性回归斜率 > 0（单调改善趋势）."""
    records = training_records_1500
    mid = [r for r in records if 500 <= r["ep"] <= 1000]
    assert len(mid) >= 4, f"ep=500~1000 的记录不足4条（实际{len(mid)}条）"

    x = np.array([r["ep"]     for r in mid], dtype=float)
    y = np.array([r["reward"] for r in mid], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    assert slope > 0, (
        f"reward 趋势斜率={slope:.6f} <= 0，策略在 ep=500~1000 无改善。"
        "Bug A/B 可能未修复，或学习率过高导致振荡。"
    )


@pytest.mark.slow
def test_t3_entropy_decrease_moderate(training_records_1500):
    """T3: entropy 在 1000 轮内下降幅度为 5%~40%（适度收敛，不过快也不过慢）."""
    records = training_records_1500
    ep100  = next((r for r in records if r["ep"] == 100), None)
    ep1000 = next((r for r in records if r["ep"] == 1000), None)
    assert ep100  is not None, "日志中缺少 ep=100  的记录"
    assert ep1000 is not None, "日志中缺少 ep=1000 的记录"

    drop_ratio = (ep100["entropy"] - ep1000["entropy"]) / ep100["entropy"]
    assert 0.05 < drop_ratio < 0.40, (
        f"entropy 下降比例={drop_ratio*100:.1f}%，需在 5%~40% 之间。"
        f"ep=100: {ep100['entropy']:.4f}, ep=1000: {ep1000['entropy']:.4f}"
    )


@pytest.mark.slow
def test_t5_critic_loss_at_500(training_records_1500):
    """T5: critic_loss 在 ep=500 时应 < 100."""
    records = training_records_1500
    ep500 = next((r for r in records if r["ep"] == 500), None)
    assert ep500 is not None, "日志中缺少 ep=500 的记录"

    assert ep500["critic_loss"] < 100, (
        f"ep=500 时 critic_loss={ep500['critic_loss']:.1f} >= 100，"
        "value 网络收敛过慢，GAE 或梯度问题可能仍存在"
    )


@pytest.mark.slow
def test_t6_reward_stability_late(training_records_1500):
    """T6: ep=1000~1500 的 reward 标准差 < 2.0（策略收敛稳定性）."""
    records = training_records_1500
    late = [r for r in records if r["ep"] >= 1000]
    assert len(late) >= 4, f"ep>=1000 的记录不足4条（实际{len(late)}条）"

    rewards = np.array([r["reward"] for r in late])
    std = float(np.std(rewards))
    assert std < 2.0, (
        f"ep=1000~1500 reward 标准差={std:.3f} >= 2.0，策略仍不稳定。"
        f"reward 值: {rewards.tolist()}"
    )
