import json
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cookierun_bot.policies import condition, learned
from cookierun_bot.policies.learned import build_convs, build_net_from_meta, LearnedAgent

FILM_META = {
    "classes": ["none", "jump", "slide"],
    "arch": "small_cnn_film",
    "K": 4, "H": 32, "W": 64,
    "crop": [0.10, 0.20, 1.00, 0.90],
    "fps": 60.0,
    "conv": [[8, 5, 2], [12, 3, 2], [16, 3, 2], [16, 3, 2]],
    "fc": 32,
    "win_pre": 0.2, "win_post": 0.03,
    "notyet_lo": 0.2, "notyet_hi": 0.65, "notyet_w": 2.5,
    "corr_w": 5.0,
    "cond": {"dims": ["t", "speed", "bonus"], "t_norm_s": 600.0,
             "speed_norm": 400.0, "bonus_latch_s": 3.0},
}


def test_film_forward_shape_and_grad():
    net = build_net_from_meta(torch, FILM_META)
    x = torch.rand(5, 4, 32, 64)
    cond = torch.rand(5, 3)
    out = net(x, cond)
    assert out.shape == (5, 3)
    out.sum().backward()          # trainable end to end


def test_film_zero_init_is_cond_invariant_at_start():
    # FiLM gamma/beta are zero-init -> BEFORE training, changing cond must not change the
    # conv features; only the (randomly-init) cond_emb path may differ. Verify the FiLM
    # modulation itself is identity: film(cond) == 0 for any cond.
    net = build_net_from_meta(torch, FILM_META)
    gb = net.film(torch.rand(7, 3))
    assert torch.allclose(gb, torch.zeros_like(gb))


def test_encoder_state_dict_transfers_between_trunk_and_film():
    import torch.nn as nn
    convs, c, h, w = build_convs(nn, FILM_META)
    net = build_net_from_meta(torch, FILM_META)
    net.convs.load_state_dict(convs.state_dict())          # strict: keys must match 1:1
    # and the same trunk state loads into the plain small_cnn prefix (shared indexing)
    plain = dict(FILM_META, arch="small_cnn")
    plain.pop("cond")
    seq = build_net_from_meta(torch, plain)
    res = seq.load_state_dict(convs.state_dict(), strict=False)
    assert not res.unexpected_keys


def test_learned_agent_film_end_to_end(tmp_path):
    net = build_net_from_meta(torch, FILM_META)
    model_p = tmp_path / "m.pt"
    meta_p = tmp_path / "m.json"
    torch.save(net.state_dict(), model_p)
    json.dump(FILM_META, open(meta_p, "w"))
    cfg = types.SimpleNamespace(
        gestures=types.SimpleNamespace(slide_conf=0.6, jump_cooldown_s=0.3,
                                       slide_hold_ms=500),
        templates_dir=str(tmp_path))               # no banner template -> bonus soft-off
    agent = LearnedAgent(cfg, str(model_p), str(meta_p), conf=0.6)
    frame = np.random.default_rng(0).integers(0, 255, (270, 480, 3), np.uint8)
    d1 = agent.decide(frame)
    d2 = agent.decide(np.roll(frame, -4, axis=1))
    assert d1.action in (0, 1, 2) and d2.action in (0, 1, 2)
    agent.reset()
    assert agent._cond._t0 is None                  # tracker reset with the agent


def test_plain_small_cnn_checkpoints_still_load():
    # regression: the build_convs refactor must not change the plain small_cnn state_dict
    plain = dict(FILM_META, arch="small_cnn")
    plain.pop("cond")
    net = build_net_from_meta(torch, plain)
    keys = list(net.state_dict().keys())
    assert keys[0] == "0.weight" and any(k.startswith("9.") or k.startswith("11.")
                                         for k in keys)


def test_continuous_slide_predictions_remain_slide(monkeypatch):
    agent = LearnedAgent.__new__(LearnedAgent)
    agent._torch = torch
    agent._stack = lambda frame: np.zeros((1, 1, 1, 1), np.float32)
    agent._device = torch.device("cpu")
    agent._cond_meta = None
    agent._net = lambda x: torch.tensor([[0.0, 0.0, 10.0]])
    agent.classes = ["none", "jump", "slide"]
    agent._conf = agent._conf_slide = 0.6
    agent.explore = 0.0
    agent._cd_until = 0.0
    times = iter([1.0, 1.1])
    monkeypatch.setattr(learned.time, "monotonic", lambda: next(times))

    assert agent.decide(None).action == 2
    assert agent.decide(None).action == 2


def test_passive_observe_advances_film_run_clock(monkeypatch):
    agent = LearnedAgent.__new__(LearnedAgent)
    agent._cond_meta = {"dims": ["t", "speed", "bonus"]}
    agent._cond = condition.CondTracker(t_norm_s=100.0)
    agent._stack = lambda frame: None
    monkeypatch.setattr(learned.time, "monotonic", lambda: 10.0)

    agent.observe(None)

    assert agent._cond.vector(20.0)[0] == pytest.approx(0.1)
