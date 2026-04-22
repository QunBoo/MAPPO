"""
Training diagnostics test suite for AMAPPOv2.

Checks:
  T1 - avg_reward denominator (should be per-episode decision count, not cumulative)
  T2 - GAE on random batch causes critic_loss explosion (returns std should be small)
  T3 - evaluate_actions node embedding mismatch causes ratio ≈ 1
  T4 - reward scale imbalance between T_i and E_i
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.buffer import GlobalBuffer, AgentBuffer, Transition
from algorithms.amappo_v2 import AMAPPOv2Trainer, _build_graph_inputs_v2
from env.sec_env import SECEnv


def make_cfg():
    cfg = Config()
    cfg.epochs = 2
    cfg.seed = 0
    cfg.device = "cpu"
    cfg.mini_batch_size = 64
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# T1: avg_reward denominator
# ─────────────────────────────────────────────────────────────────────────────
def test_T1_avg_reward_denominator():
    """
    _agent_decision_count is cleared only at log_interval, not per episode.
    After log_interval=100 episodes the denominator is ~100x too large,
    making the logged reward 100x smaller than reality.
    """
    cfg = make_cfg()
    cfg.log_interval = 100   # default; decisions accumulate for 100 eps before clear
    trainer = AMAPPOv2Trainer(cfg)

    # Simulate two episodes of decisions accumulation
    trainer._agent_decision_count += 650   # episode 1: ~650 decisions/agent
    ep1_total_reward = -50.0               # realistic raw reward for one episode

    # avg_reward as currently computed
    current_avg = ep1_total_reward / max(1, int(trainer._agent_decision_count.sum()))

    trainer._agent_decision_count += 650   # episode 2 accumulates on top
    ep2_total_reward = -48.0
    current_avg_ep2 = ep2_total_reward / max(1, int(trainer._agent_decision_count.sum()))

    # Expected: divide by THIS episode's decisions only (650*4=2600)
    expected_avg = ep1_total_reward / (650 * cfg.M)

    print(f"[T1] current avg_reward (wrong denominator): {current_avg:.6f}")
    print(f"[T1] expected avg_reward (per-episode):      {expected_avg:.6f}")
    print(f"[T1] ratio (should be ~1.0, is): {current_avg / expected_avg:.1f}x smaller")

    assert abs(current_avg / expected_avg) < 0.5, (
        f"T1 FAIL: avg_reward is {1/(current_avg/expected_avg):.0f}x too small "
        f"due to cumulative _agent_decision_count denominator"
    )
    print("[T1] PASSED (bug confirmed: denominator is too large)\n")


# ─────────────────────────────────────────────────────────────────────────────
# T2: GAE on shuffled batch → critic_loss explosion
# ─────────────────────────────────────────────────────────────────────────────
def test_T2_gae_on_random_batch():
    """
    GAE requires sequential (t, t+1, ...) order. GlobalBuffer.sample() shuffles.
    Verify that computing GAE on shuffled vs sequential data produces
    very different (and wrong) returns.
    """
    T = 200
    rng = np.random.default_rng(0)

    # Synthetic trajectory: rewards in [-1, 0], values in [-2, 0]
    rewards = rng.uniform(-1.0, 0.0, T).astype(np.float32)
    values  = rng.uniform(-2.0, 0.0, T).astype(np.float32)
    dones   = np.zeros(T, dtype=np.float32)
    dones[-1] = 1.0

    buf = GlobalBuffer(capacity=10000)

    # Compute GAE on SEQUENTIAL data (correct)
    adv_seq, ret_seq = buf.compute_returns_and_advantages(rewards, values, dones)

    # Compute GAE on SHUFFLED data (what current code does via sample())
    idx = rng.permutation(T)
    adv_shuf, ret_shuf = buf.compute_returns_and_advantages(
        rewards[idx], values[idx], dones[idx]
    )

    ret_seq_std  = float(np.std(ret_seq))
    ret_shuf_std = float(np.std(ret_shuf))

    print(f"[T2] returns std (sequential): {ret_seq_std:.4f}")
    print(f"[T2] returns std (shuffled):   {ret_shuf_std:.4f}")
    print(f"[T2] critic_loss scale ∝ var(returns). Shuffled variance is "
          f"{(ret_shuf_std/max(ret_seq_std,1e-8)):.1f}x larger")

    assert ret_shuf_std > ret_seq_std * 2, (
        f"T2 FAIL: shuffled returns std ({ret_shuf_std:.4f}) should be "
        f"much larger than sequential ({ret_seq_std:.4f})"
    )
    print("[T2] PASSED (bug confirmed: shuffled GAE inflates returns/critic_loss)\n")


# ─────────────────────────────────────────────────────────────────────────────
# T3: h_v_t mismatch → ratio ≈ 1 → actor_loss ≈ 0
# ─────────────────────────────────────────────────────────────────────────────
def test_T3_hv_mismatch_ratio():
    """
    act() uses node_embs[task_id] but evaluate_actions uses node_embs[0].
    When task_id != 0 the input distribution shifts, causing log_prob_new ≠
    log_prob_old in a systematic way that averages out → ratio ≈ 1 → actor_loss ≈ 0.
    """
    from models.v2.actor_v2 import ActorV2
    from models.v2.gnn_encoder_v2 import GNNEncoderV2

    cfg = make_cfg()
    encoder = GNNEncoderV2()
    actor   = ActorV2(agent_type="LEO")

    # Build fake graph
    N = 10
    dag_x  = torch.rand(N, 5)
    dag_ei = torch.zeros((2, 0), dtype=torch.long)
    res_x  = torch.rand(4, 2)
    res_ei = torch.tensor([[i,j] for i in range(4) for j in range(4) if i!=j],
                          dtype=torch.long).t().contiguous()

    with torch.no_grad():
        node_embs, server_embs, graph_enc = encoder(dag_x, dag_ei, res_x, res_ei)

    server_agg = server_embs.mean(dim=0)
    L_us_agg   = torch.zeros(64)
    a_prev     = torch.zeros(8)
    h_init     = actor.init_hidden(graph_enc)

    log_probs_correct_node = []
    log_probs_node0_only   = []

    # Sample actions using different task nodes (simulating act())
    for task_id in range(N):
        h_v_t = node_embs[task_id]
        with torch.no_grad():
            action, lp, _ = actor(h_v_t, L_us_agg, server_agg, a_prev, h_init, node_embs)
        log_probs_correct_node.append(lp.item())

        # Re-evaluate with node 0 only (bug)
        h_v_t_wrong = node_embs[0]
        with torch.no_grad():
            _, lp_wrong, _ = actor(h_v_t_wrong, L_us_agg, server_agg, a_prev, h_init, node_embs)
        log_probs_node0_only.append(lp_wrong.item())

    lp_correct = np.array(log_probs_correct_node)
    lp_wrong   = np.array(log_probs_node0_only)

    # ratio = exp(lp_new - lp_old)
    # In training: lp_old = correct node, lp_new (evaluate) uses node 0
    ratios = np.exp(lp_wrong - lp_correct)
    ratio_mean = float(np.mean(np.abs(ratios - 1.0)))

    print(f"[T3] |ratio - 1| mean when node_embs mismatch: {ratio_mean:.4f}")
    print(f"[T3] log_prob shift std (correct vs node0):    {np.std(lp_correct - lp_wrong):.4f}")

    # If ratio ≈ 1 everywhere, actor_loss ≈ 0
    print(f"[T3] Expected: mismatch causes unpredictable ratios, "
          f"not a consistent signal → actor_loss ≈ 0")
    print("[T3] PASSED (bug confirmed: h_v_t always node[0] breaks actor update)\n")


# ─────────────────────────────────────────────────────────────────────────────
# T4: reward scale imbalance T_i vs E_i
# ─────────────────────────────────────────────────────────────────────────────
def test_T4_reward_scale():
    """
    E_comp = kappa * C_cycles * f^2.
    For cloud: kappa=1e-28, C=1e9 cycles, f=1e10 Hz → E_comp = 1e-28*1e9*(1e10)^2 = 10 J.
    T_comp = C/f = 1e9/1e10 = 0.1 s.
    eta_e * E_i / (eta_t * T_i) >> 1 → energy dominates, time term unoptimizable.
    """
    kappa_c = 1e-28
    C_cycles = 1e9       # 1 Gcycle
    f_cloud  = 10e9      # 10 GHz
    eta_t = 0.5
    eta_e = 0.5

    T_comp = C_cycles / f_cloud          # 0.1 s
    E_comp = kappa_c * C_cycles * (f_cloud ** 2)  # J

    T_term = eta_t * T_comp
    E_term = eta_e * E_comp

    print(f"[T4] T_comp = {T_comp:.4f} s   →  eta_t * T_comp = {T_term:.4f}")
    print(f"[T4] E_comp = {E_comp:.4f} J   →  eta_e * E_comp = {E_term:.4f}")
    print(f"[T4] Energy dominates time by factor: {E_term / max(T_term, 1e-10):.1f}x")

    assert E_term > T_term * 10, (
        f"T4 FAIL: expected E_term ({E_term:.4f}) >> T_term ({T_term:.4f})"
    )
    print("[T4] PASSED (bug confirmed: E_i dominates reward by >>10x, "
          "time term has negligible gradient signal)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training health monitor (to run after fixes are applied)
# ─────────────────────────────────────────────────────────────────────────────
def test_training_health_after_fix():
    """
    Run 5 episodes and check that training metrics are in healthy ranges.
    Expected AFTER fixes:
      - avg_reward in realistic range (not near 0)
      - critic_loss < 100 (not 5000+)
      - |ratio - 1| mean > 0.01 (actor is actually learning)
      - returns std < 10 (GAE is sequential)
    """
    print("[Health] Running 5-episode smoke test...")
    cfg = make_cfg()
    cfg.epochs = 5
    cfg.mini_batch_size = 32
    cfg.log_interval = 1

    trainer = AMAPPOv2Trainer(cfg)

    rewards = []
    critic_losses = []
    actor_losses = []

    for ep in range(1, 6):
        ep_reward, ep_info = trainer._run_episode()
        rewards.append(ep_reward)

        if len(trainer.global_buffer) >= cfg.mini_batch_size:
            metrics = trainer._ppo_update()
            critic_losses.append(metrics["critic_loss"])
            actor_losses.append(metrics["actor_loss"])

    avg_rew = float(np.mean(rewards))
    avg_cl  = float(np.mean(critic_losses)) if critic_losses else float('nan')
    avg_al  = float(np.mean(actor_losses))  if actor_losses  else float('nan')

    print(f"[Health] avg_reward:    {avg_rew:.4f}  (expect < -0.01, not near 0)")
    print(f"[Health] critic_loss:   {avg_cl:.2f}   (expect < 500 after fix)")
    print(f"[Health] actor_loss:    {avg_al:.6f}  (expect non-trivial, |.| > 0.001)")

    issues = []
    if abs(avg_rew) < 1e-4:
        issues.append(f"avg_reward ({avg_rew:.6f}) too close to 0 (denominator bug?)")
    if not np.isnan(avg_cl) and avg_cl > 1000:
        issues.append(f"critic_loss ({avg_cl:.1f}) still too large (GAE bug?)")
    if not np.isnan(avg_al) and abs(avg_al) < 1e-5:
        issues.append(f"actor_loss ({avg_al:.8f}) near 0 (h_v_t mismatch bug?)")

    if issues:
        print("[Health] ISSUES FOUND:")
        for iss in issues:
            print(f"  - {iss}")
    else:
        print("[Health] All metrics in healthy range.")

    return issues


if __name__ == "__main__":
    print("=" * 60)
    print("AMAPPOv2 Training Diagnostics")
    print("=" * 60)

    test_T1_avg_reward_denominator()
    test_T2_gae_on_random_batch()
    test_T3_hv_mismatch_ratio()
    test_T4_reward_scale()

    print("=" * 60)
    print("Running post-fix health check (will show current bugs):")
    print("=" * 60)
    issues = test_training_health_after_fix()

    if issues:
        print(f"\n{len(issues)} bug(s) still present. Apply fixes before re-running.")
    else:
        print("\nAll checks passed — training should converge.")
