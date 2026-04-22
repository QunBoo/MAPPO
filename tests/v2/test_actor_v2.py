import pytest
import torch
from models.v2.actor_v2 import ActorV2


def _make_inputs(n_tasks=5, batch=None):
    if batch is None:
        h_v_t = torch.randn(64)
        L_us_agg = torch.randn(64)
        server_agg = torch.randn(64)
        a_prev = torch.randn(8)
        h_prev = None
        node_embs = torch.randn(n_tasks, 64)
    else:
        h_v_t = torch.randn(batch, 64)
        L_us_agg = torch.randn(batch, 64)
        server_agg = torch.randn(batch, 64)
        a_prev = torch.randn(batch, 8)
        h_prev = torch.zeros(1, batch, 64)
        node_embs = torch.randn(n_tasks, 64)
    return h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs


def test_leo_forward_shapes():
    actor = ActorV2(agent_type="LEO")
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs()

    action, log_prob, h_next = actor(h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs)

    assert action.shape == (7,), f"LEO action: {action.shape}"
    assert log_prob.shape == (), f"log_prob: {log_prob.shape}"
    assert h_next.shape == (1, 1, 64), f"h_next: {h_next.shape}"


def test_uav_forward_shapes():
    actor = ActorV2(agent_type="UAV")
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs()

    action, log_prob, h_next = actor(h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs)

    assert action.shape == (9,), f"UAV action: {action.shape}"
    assert log_prob.shape == (), f"log_prob: {log_prob.shape}"


def test_gru_hidden_state_propagates():
    actor = ActorV2(agent_type="LEO")
    h_v_t, L_us_agg, server_agg, a_prev, _, node_embs = _make_inputs()

    _, _, h1 = actor(h_v_t, L_us_agg, server_agg, a_prev, None, node_embs)
    _, _, h2 = actor(h_v_t, L_us_agg, server_agg, a_prev, h1, node_embs)

    assert not torch.allclose(h1, h2), "Hidden state should change across steps"


def test_evaluate_leo():
    actor = ActorV2(agent_type="LEO")
    B = 4
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs(batch=B)
    discrete_actions = torch.randint(0, 4, (B,))

    log_probs, entropies, h_next = actor.evaluate(
        h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs, discrete_actions
    )

    assert log_probs.shape == (B,)
    assert entropies.shape == (B,)
    assert h_next.shape == (1, B, 64)


def test_evaluate_uav():
    actor = ActorV2(agent_type="UAV")
    B = 3
    h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs = _make_inputs(batch=B)
    discrete_actions = torch.randint(0, 4, (B,))

    log_probs, entropies, h_next = actor.evaluate(
        h_v_t, L_us_agg, server_agg, a_prev, h_prev, node_embs, discrete_actions
    )

    assert log_probs.shape == (B,)
    assert entropies.shape == (B,)


def test_gradient_flow():
    actor = ActorV2(agent_type="LEO")
    h_v_t, L_us_agg, server_agg, a_prev, _, node_embs = _make_inputs()

    action, log_prob, h_next = actor(h_v_t, L_us_agg, server_agg, a_prev, None, node_embs)
    log_prob.backward()

    # hidden_init is only used in init_hidden(), not in forward(), so it won't have grads here
    skip_params = {"hidden_init.weight", "hidden_init.bias"}
    for name, p in actor.named_parameters():
        if p.requires_grad and name not in skip_params:
            assert p.grad is not None, f"No gradient for {name}"


def test_invalid_agent_type():
    with pytest.raises(ValueError, match="agent_type"):
        ActorV2(agent_type="INVALID")
