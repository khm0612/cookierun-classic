"""Imitation-learned dodge policy: a CNN (trained by behavioral cloning on the user's own
play, scripts/train2.py) predicts jump / slide / none from a stack of recent frames.
Drop-in for the rule-based agent — exposes .decide(frame) -> ActionDecision and .act(frame).

Everything that must match training EXACTLY (architecture, crop, input size, frame-stack
temporal spacing) is driven by model_meta.json, so training and inference cannot drift.
Loaded lazily so importing this module never requires torch unless a LearnedAgent is built.
"""
from __future__ import annotations
from collections import deque
import json
import time
import cv2
import numpy as np

from ..gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE
from .rule_based import ActionDecision

_ACTION = {"none": ACTION_NOOP, "jump": ACTION_JUMP, "slide": ACTION_SLIDE}


def _replace_first_conv(nn, module, in_ch: int) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, nn.Conv2d(
                in_ch, child.out_channels, child.kernel_size, child.stride, child.padding,
                child.dilation, child.groups, child.bias is not None, child.padding_mode))
            return
        try:
            _replace_first_conv(nn, child, in_ch)
            return
        except LookupError:
            pass
    raise LookupError("no Conv2d layer found")


def _torchvision_model(model_name: str, num_classes: int):
    try:
        from torchvision import models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{model_name} requires torchvision. Install it with "
            "`python -m pip install torchvision`."
        ) from exc
    return getattr(models, model_name)(weights=None, num_classes=num_classes)


def _build_mobilenet_v3_large(torch, meta):
    import torch.nn as nn
    net = _torchvision_model("mobilenet_v3_large", len(meta["classes"]))
    _replace_first_conv(nn, net.features, int(meta["K"]))
    return net


def _build_efficientnet_b5(torch, meta):
    import torch.nn as nn
    net = _torchvision_model("efficientnet_b5", len(meta["classes"]))
    _replace_first_conv(nn, net.features, int(meta["K"]))
    return net


def build_convs(nn, meta):
    """The conv trunk shared by small_cnn, FilmCNN and the SSL pretrainer (scripts/
    pretrain_encoder.py): Sequential of Conv2d+ReLU from meta['conv'] with the SAME layer
    indexing as the small_cnn prefix, so a pretrained encoder state_dict loads 1:1 into
    any of them. Returns (convs, out_channels, out_h, out_w)."""
    layers, in_ch = [], meta["K"]
    h, w = meta["H"], meta["W"]
    for out_ch, k, s in meta["conv"]:
        layers += [nn.Conv2d(in_ch, out_ch, k, s, k // 2), nn.ReLU()]
        in_ch = out_ch
        h, w = (h + s - 1) // s, (w + s - 1) // s
    return nn.Sequential(*layers), in_ch, h, w


def _build_film_cnn(torch, meta):
    """small_cnn trunk + FiLM conditioning on [t, speed, bonus] (see condition.py). The
    cond vector modulates the last conv feature map per-channel (gamma/beta, ZERO-init so
    an untrained FiLM is an exact identity = stable fine-tune from a pretrained encoder)
    and is also embedded + concatenated into the fc input."""
    import torch.nn as nn
    cond_dim = len(meta.get("cond", {}).get("dims", ["t", "speed", "bonus"]))
    convs, c, h, w = build_convs(nn, meta)
    fc_dim, n_cls = meta["fc"], len(meta["classes"])

    class FilmCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = convs
            self.film = nn.Sequential(nn.Linear(cond_dim, 32), nn.ReLU(),
                                      nn.Linear(32, 2 * c))
            self.cond_emb = nn.Sequential(nn.Linear(cond_dim, 16), nn.ReLU())
            self.fc = nn.Linear(c * h * w + 16, fc_dim)
            self.drop = nn.Dropout(0.3)
            self.head = nn.Linear(fc_dim, n_cls)
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

        def forward(self, x, cond):
            hh = self.convs(x)
            g, b = self.film(cond).chunk(2, dim=-1)
            hh = hh * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
            z = torch.cat([hh.flatten(1), self.cond_emb(cond)], dim=1)
            return self.head(self.drop(torch.relu(self.fc(z))))

    return FilmCNN()


def build_net_from_meta(torch, meta):
    """Build the architecture named in model_meta.json. Missing arch keeps old checkpoints
    on the original small CNN."""
    arch = meta.get("arch", "small_cnn")
    if arch == "mobilenet_v3_large":
        return _build_mobilenet_v3_large(torch, meta)
    if arch == "efficientnet_b5":
        return _build_efficientnet_b5(torch, meta)
    if arch == "small_cnn_film":
        return _build_film_cnn(torch, meta)
    if arch not in ("small_cnn", "cnn", "conv"):
        raise ValueError(f"unknown learned model arch: {arch}")

    # Conv stack from meta['conv'] = [(out_ch, kernel, stride), ...] on K stacked grayscale
    # frames, flattened WITH spatial layout preserved (no global pooling — obstacle POSITION
    # is the signal), then fc -> 3 classes. Shared by train2.py and LearnedAgent.
    import torch.nn as nn
    convs, in_ch, h, w = build_convs(nn, meta)
    layers = list(convs)
    layers += [nn.Flatten(), nn.Linear(in_ch * h * w, meta["fc"]), nn.ReLU(),
               nn.Dropout(0.3), nn.Linear(meta["fc"], len(meta["classes"]))]
    return nn.Sequential(*layers)


class LearnedAgent:
    """CNN behavioral-cloning policy. `conf` = minimum softmax probability to act (below it
    -> NOOP): trades a few missed dodges for far fewer spurious ones.

    `conf_slide` = the slide-specific gate. USER CORRECTION (2026-07-06): sliding is CHEAP — a
    wrong/low-confidence slide does NOT kill the cookie. The only cost of over-sliding is that a
    held slide BLOCKS the one-finger jump, so spamming slide can miss a needed jump; a single
    well-placed slide is free. So conf_slide is now LOW (from cfg.gestures.slide_conf, default
    0.60) — the model should DUCK READILY — not the old near-certain 0.90 that (wrongly) assumed
    a mistaken slide caused a pit death. Pass conf_slide explicitly to override the config."""

    def __init__(self, cfg, model_path: str, meta_path: str, conf: float = 0.6,
                 conf_slide: float | None = None):
        import torch, os
        self._torch = torch
        # Allow forcing GPU-only mode via env var: set FORCE_CUDA=1 to require CUDA and fail otherwise.
        force_cuda = os.environ.get("FORCE_CUDA", os.environ.get("CUDA_ONLY", "0"))
        force_cuda = str(force_cuda).lower() in ("1", "true", "yes")
        if force_cuda and not torch.cuda.is_available():
            raise RuntimeError("FORCE_CUDA is set but no CUDA device is available. Aborting to avoid CPU fallback.")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta = json.load(open(meta_path))
        self.meta = meta
        self.K, self.H, self.W = meta["K"], meta["H"], meta["W"]
        self.classes = meta["classes"]
        self._crop = meta.get("crop", [0.0, 0.0, 1.0, 1.0])
        # stack frames at the TRAINING fps spacing: live capture may run far faster (dxcam
        # ~270fps) and a 15ms-span stack would be out-of-distribution. meta['fps'] is now
        # MEASURED from the demos' real frame cadence at train time (scripts/train2.py), so
        # it tracks the recorder instead of a stale hardcoded assumption.
        self._frame_gap = 1.0 / meta.get("fps", 35.0)
        self._last_stacked = 0.0
        self._conf = conf
        self._conf_slide = conf_slide if conf_slide is not None else \
            getattr(getattr(cfg, "gestures", None), "slide_conf", 0.60)
        self._buf: deque = deque(maxlen=self.K)
        # FiLM conditioning (arch small_cnn_film): meta["cond"] carries the normalisation
        # constants; the agent computes the SAME [t, speed, bonus] vector live that
        # scripts/train2.py computed offline from the demos. Self-contained: the BONUSTIME
        # banner is checked here (throttled), so ai_farm/self_farm/farm all get it for
        # free. The template is machine-local — absent => bonus soft-off at 0.
        self._cond_meta = meta.get("cond") if meta.get("arch") == "small_cnn_film" else None
        if meta.get("arch") == "small_cnn_film" and not self._cond_meta:
            raise ValueError("small_cnn_film checkpoint has no 'cond' in its meta — the "
                             "forward pass needs it; retrain with current train2.py")
        if self._cond_meta:
            from .condition import CondTracker, bonustime_bgr, load_bonus_template
            if not self._cond_meta.get("speed_norm"):
                raise ValueError("film meta cond.speed_norm is missing/0 — the speed dim "
                                 "would silently pin at the clip; retrain with train2.py")
            self._bonustime_bgr = bonustime_bgr
            self._cond = CondTracker(
                t_norm_s=self._cond_meta.get("t_norm_s", 600.0),
                speed_norm=self._cond_meta["speed_norm"],
                bonus_latch_s=self._cond_meta.get("bonus_latch_s", 3.0),
                scroll_v=self._cond_meta.get("scroll_v", 1))
            # bonus_trained=False => the model learned with the bonus dim all-0 (template
            # was absent at train time) — feed the SAME all-0 live even if this machine
            # has the template, or the model would see an input it never trained on.
            self._bt_tpl = None
            if self._cond_meta.get("bonus_trained", True):
                self._bt_tpl = load_bonus_template(getattr(cfg, "templates_dir", "templates"))
                if self._bt_tpl is None:
                    print("[learned] WARNING: model trained WITH the bonus dim but "
                          "templates/bonustime_norm.png is missing here — bonus stuck 0")
            self._bt_check_s = 0.25
            self._bt_last = 0.0
            self._newest_img_t = 0.0    # capture time of the image in the newest slot
        self._net = build_net_from_meta(torch, meta)
        self._net.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._net.to(self._device).eval()
        # TIME-based jump cooldown (tick-based breaks at 100+fps live loops). Small on
        # purpose: the human demo double-jumps with gaps down to ~0.12s (p5 0.20s), so a
        # long cooldown blocks half their real dodge pattern; the device's one-finger
        # throttle already absorbs per-frame refires of the same decision.
        # TIME-based jump cooldown (config-tunable). Bigger = fewer, slower jumps (the user asked
        # to "slow down the jump a bit"). Small enough to still allow the human's fast double-jumps.
        self._jump_cd_s = getattr(getattr(cfg, "gestures", None), "jump_cooldown_s", 0.30)
        self._cd_until = 0.0
        # EXPLORATION (self-improvement, no labels): >0 samples non-greedy actions at UNCERTAIN
        # frames so a self-farm run can stumble into better timings; survival then keeps the good
        # ones. 0 = pure greedy (banking). Set per-run by the farm; confident dodges stay greedy.
        self.explore = 0.0
        # Slide gets its OWN cooldown ~= its hold duration. play_until_death re-decides per
        # frame and each SLIDE re-issues device.hold(...slide_hold_ms) fire-and-forget, so an
        # ungated high-conf slide at 60-270fps queues many overlapping holds = seconds of
        # continuous LOW posture (over-slides through a platform gap = pit death) + an adb
        # backlog. Don't re-issue a slide while the previous hold is still in flight.
        self._slide_cd_s = getattr(getattr(cfg, "gestures", None), "slide_hold_ms", 500) / 1000.0
        self._slide_cd_until = 0.0

    def reset(self) -> None:
        self._buf.clear()
        self._cd_until = 0.0
        self._slide_cd_until = 0.0
        self._last_stacked = 0.0
        if self._cond_meta:
            self._cond.reset()
            self._newest_img_t = 0.0

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._crop
        band = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (self.W, self.H), interpolation=cv2.INTER_AREA)

    def _stack(self, frame):
        now = time.monotonic()
        if not self._buf or now - self._last_stacked >= self._frame_gap:
            new = self._preprocess(frame).astype(np.float32) / 255.0
            if self._cond_meta and self._buf:
                # one speed sample per NEW slot = the offline per-recorded-frame cadence.
                # dt must be the age of the IMAGE in the newest slot, not the slot gap:
                # sub-gap ticks refresh that slot with fresher pixels, so dividing their
                # motion by the full slot gap under-reads speed by ~meta_fps/loop_fps.
                self._cond.on_slot(self._buf[-1], new, now - self._newest_img_t)
            self._buf.append(new)
            self._last_stacked = now
            if self._cond_meta:
                self._newest_img_t = now
        else:                                   # too soon: refresh only the newest slot
            self._buf[-1] = self._preprocess(frame).astype(np.float32) / 255.0
            if self._cond_meta:
                self._newest_img_t = now
        while len(self._buf) < self.K:
            self._buf.appendleft(self._buf[0])
        return np.stack(self._buf, 0)[None]     # (1,K,H,W)

    def observe(self, frame) -> None:
        """Update the frame stack (and cond tracker) WITHOUT running inference — lets a
        wrapper keep an idle model's temporal state warm so a hand-off is never fed a
        stale/duplicated K-stack (policies/hybrid_phase.py)."""
        self._stack(frame)

    def decide(self, frame) -> ActionDecision:
        x = self._torch.from_numpy(self._stack(frame)).to(self._device)
        if self._cond_meta:
            tnow = time.monotonic()
            if tnow - self._bt_last >= self._bt_check_s:   # throttled banner check
                self._bt_last = tnow
                if self._bonustime_bgr(frame, self._bt_tpl):
                    self._cond.bonus_seen(tnow)
            cond_t = self._torch.from_numpy(self._cond.vector(tnow)[None]).to(self._device)
            with self._torch.no_grad():
                p = self._torch.softmax(self._net(x, cond_t)[0], 0).cpu().numpy()
        else:
            with self._torch.no_grad():
                p = self._torch.softmax(self._net(x)[0], 0).cpu().numpy()
        i = int(p.argmax())
        # EXPLORE only where the policy is UNSURE (top prob < 0.85): sample the action from p and
        # COMMIT to it (bypass the conf gate) so the alternative is actually tried. Confident dodges
        # (>=0.85) are never randomised, so exploration can't blow up a clear obstacle response.
        explored = False
        if self.explore > 0.0 and float(p.max()) < 0.85 and np.random.random() < self.explore:
            i = int(np.random.choice(len(p), p=p))
            explored = True
        cls = self.classes[i]
        action = _ACTION[cls]
        now = time.monotonic()
        if explored:
            if action == ACTION_NOOP:
                return ActionDecision(ACTION_NOOP, f"explore:none:{p[i]:.2f}")
            if action == ACTION_SLIDE and p[i] < self._conf_slide:
                # a wrong SLIDE keeps the cookie low through a platform gap = un-tankable pit death
                # (asymmetric vs jump, which just lands back). Never GAMBLE a slide while exploring —
                # exploration only tries jump/none timings (recoverable errors).
                return ActionDecision(ACTION_NOOP, f"explore:slide-gated:{p[i]:.2f}")
            if action == ACTION_JUMP:
                if now < self._cd_until:
                    return ActionDecision(ACTION_NOOP, "explore:jump-cooldown")
                self._cd_until = now + self._jump_cd_s
            return ActionDecision(action, f"explore:{cls}:{p[i]:.2f}")
        gate = self._conf_slide if action == ACTION_SLIDE else self._conf
        if action == ACTION_NOOP or p[i] < gate:
            return ActionDecision(ACTION_NOOP, f"model:{cls}:{p[i]:.2f}")
        if action == ACTION_JUMP:
            if now < self._cd_until:
                return ActionDecision(ACTION_NOOP, "model:jump-cooldown")
            self._cd_until = now + self._jump_cd_s
        elif action == ACTION_SLIDE:
            if now < self._slide_cd_until:
                return ActionDecision(ACTION_NOOP, "model:slide-cooldown")
            self._slide_cd_until = now + self._slide_cd_s
        return ActionDecision(action, f"model:{cls}:{p[i]:.2f}")

    def act(self, frame) -> int:
        return self.decide(frame).action
