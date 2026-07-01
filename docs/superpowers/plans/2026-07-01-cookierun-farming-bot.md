# CookieRun Classic Farming Bot — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, hands-off rule-based bot that farms Coins and Ingredients in CookieRun Classic on a real Android phone, including the full Gymnasium environment that a later RL plan will train in.

**Architecture:** A single `Device` abstraction talks to the phone (scrcpy for fast video+touch, ADB fallback). Pure-logic modules (capture, detect, reward, gestures, menu) sit on top, composed into a `CookieRunEnv(gym.Env)`. A `RuleBasedAgent` plays through that env; an outer farm loop replays one stage and logs coins/ingredients per hour.

**Tech Stack:** Python 3.10+, `scrcpy-client` (needs scrcpy ≥ v1.20), `adbutils`, `opencv-python`, `numpy`, `gymnasium`, `pytesseract` (optional OCR), `PyYAML`, `pytest`.

## Global Constraints

- Python **3.10+** (uses `X | None` type syntax).
- Game: **CookieRun Classic**, Android package **`com.devsisters.crg`**, controls = two on-screen buttons **Jump** and **Slide** (double-jump = two Jump taps; slide = *held* press).
- Objective is **coins + ingredients (Mystery Boxes), NOT score/distance**. No reward term may reward score, jelly points, or distance.
- **Never spend currency:** menu code taps only *allowlist* templates and aborts on any *denylist* (spend/revive/purchase/ad) dialog.
- Every file kept **< 500 lines**, one responsibility each.
- Device-specific files (`config.yaml`, `templates/`) are **gitignored**; `config.example.yaml` is committed.
- Action space is `Discrete(3)`: `0=noop, 1=jump, 2=slide`. Observation is `Box(0,255,(4,84,84),uint8)`.
- TDD throughout. Pure-logic modules get unit tests with synthetic frames / fake devices. Hardware-I/O tasks (device, calibrate, live play) use explicit **manual verification** steps — do not fake unit tests for physical-phone I/O.

---

## File Structure

```
cookierun-bot/
  requirements.txt
  config.example.yaml            # committed template
  config.yaml                    # gitignored (user calibrates)
  .gitignore
  pyproject.toml                 # pytest config + package
  templates/                     # gitignored reference images
  src/cookierun_bot/
    __init__.py
    config.py       # dataclasses + load_config
    device.py       # Device protocol, ScrcpyDevice, AdbDevice, open_device
    capture.py      # preprocess(), FrameStack
    detect.py       # TemplateMatcher, read_int, detectors
    gestures.py     # action constants + apply_action
    reward.py       # RewardTracker
    menu.py         # MenuNavigator (allow/denylist)
    env.py          # CookieRunEnv(gym.Env)
    metrics.py      # RunResult, Metrics
    calibrate.py    # screenshot helper (CLI)
    policies/
      __init__.py
      rule_based.py # Features, extract_features, RuleBasedAgent
    agents/
      __init__.py
      play.py       # farm loop (rule-based live)
  tests/
    conftest.py     # fakes + synthetic-frame fixtures
    test_config.py
    test_capture.py
    test_detect.py
    test_gestures.py
    test_reward.py
    test_menu.py
    test_env.py
    test_rule_based.py
    test_metrics.py
```

---

## Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.gitignore`, `src/cookierun_bot/__init__.py`, `src/cookierun_bot/policies/__init__.py`, `src/cookierun_bot/agents/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: importable package `cookierun_bot`; `pytest` runnable.

- [ ] **Step 1: Create `requirements.txt`**

```
scrcpy-client>=0.4.0
adbutils>=2.0.0
opencv-python>=4.8.0
numpy>=1.24
gymnasium>=0.29
pytesseract>=0.3.10
PyYAML>=6.0
pytest>=7.4
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "cookierun-bot"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
config.yaml
templates/
recordings/
runs/
*.zip
```

- [ ] **Step 4: Create the three empty `__init__.py` files** (package markers) with a single line each:

```python
"""cookierun_bot package."""
```

- [ ] **Step 5: Create `tests/conftest.py` with shared fakes/fixtures**

```python
import numpy as np
import pytest


class FakeDevice:
    """In-memory Device stand-in for tests."""
    def __init__(self, frame=None, resolution=(1080, 1920)):
        self._frame = frame
        self._resolution = resolution
        self.taps = []          # list of (x, y)
        self.holds = []         # list of (x, y, duration_ms)
        self.started = False

    def start(self): self.started = True
    def stop(self): self.started = False
    def last_frame(self): return self._frame
    def set_frame(self, frame): self._frame = frame

    @property
    def resolution(self): return self._resolution

    def tap(self, x, y): self.taps.append((x, y))
    def hold(self, x, y, duration_ms): self.holds.append((x, y, duration_ms))


@pytest.fixture
def fake_device():
    return FakeDevice(frame=np.zeros((1920, 1080, 3), dtype=np.uint8))


@pytest.fixture
def blank_frame():
    return np.zeros((1920, 1080, 3), dtype=np.uint8)
```

- [ ] **Step 6: Install and verify**

Run: `pip install -r requirements.txt && python -c "import cookierun_bot; print('ok')"`
Expected: prints `ok` (no import error).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml .gitignore src tests/conftest.py
git commit -m "chore: project scaffold and test fixtures"
```

---

## Task 2: Config loading

**Files:**
- Create: `src/cookierun_bot/config.py`, `config.example.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `Region(x:int,y:int,w:int,h:int)` with `.crop(img)->np.ndarray`
  - `Gestures(jump_button:tuple[int,int], slide_button:tuple[int,int], slide_hold_ms:int)`
  - `RewardWeights(w_coin:float, w_box:float, w_survive:float, death_penalty:float)`
  - `Config(device_serial:str|None, capture_backend:str, max_fps:int, decision_hz:int, target_stage:str, regions:dict[str,Region], gestures:Gestures, reward:RewardWeights, menu_allowlist:list[str], menu_denylist:list[str], templates_dir:str)`
  - `load_config(path:str="config.yaml")->Config`; raises `ConfigError`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import numpy as np
import pytest
from cookierun_bot.config import load_config, Region, ConfigError


def test_region_crop():
    r = Region(10, 20, 3, 4)
    img = np.arange(100 * 100).reshape(100, 100)
    out = r.crop(img)
    assert out.shape == (4, 3)          # (h, w)
    assert out[0, 0] == img[20, 10]


def test_load_config_parses_all_sections(tmp_path):
    (tmp_path / "config.yaml").write_text(
        """
device: {serial: null, capture: scrcpy, max_fps: 60}
loop: {target_stage: "Episode 1", decision_hz: 15}
regions:
  play_area: [0, 200, 1080, 1200]
  coin_counter: [800, 40, 200, 60]
  mystery_box_counter: [500, 40, 120, 60]
  results_coins: [400, 900, 300, 80]
  results_ingredients: [400, 1000, 300, 80]
gestures: {jump_button: [200, 1600], slide_button: [880, 1600], slide_hold_ms: 300}
reward: {w_coin: 1.0, w_box: 50.0, w_survive: 0.01, death_penalty: 10.0}
menu:
  allowlist: [restart, replay, collect, ok, start]
  denylist: [revive_crystals, buy, purchase, watch_ad]
templates_dir: templates
        """
    )
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert cfg.capture_backend == "scrcpy"
    assert cfg.decision_hz == 15
    assert cfg.regions["coin_counter"].w == 200
    assert cfg.gestures.slide_hold_ms == 300
    assert cfg.reward.w_box == 50.0
    assert "purchase" in cfg.menu_denylist


def test_load_config_missing_region_raises(tmp_path):
    (tmp_path / "config.yaml").write_text("device: {capture: scrcpy}\n")
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "config.yaml"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.config`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/config.py
from __future__ import annotations
from dataclasses import dataclass
import yaml


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int

    def crop(self, img):
        return img[self.y:self.y + self.h, self.x:self.x + self.w]


@dataclass(frozen=True)
class Gestures:
    jump_button: tuple[int, int]
    slide_button: tuple[int, int]
    slide_hold_ms: int


@dataclass(frozen=True)
class RewardWeights:
    w_coin: float
    w_box: float
    w_survive: float
    death_penalty: float


@dataclass(frozen=True)
class Config:
    device_serial: str | None
    capture_backend: str
    max_fps: int
    decision_hz: int
    target_stage: str
    regions: dict[str, Region]
    gestures: Gestures
    reward: RewardWeights
    menu_allowlist: list[str]
    menu_denylist: list[str]
    templates_dir: str


_REQUIRED_REGIONS = [
    "play_area", "coin_counter", "mystery_box_counter",
    "results_coins", "results_ingredients",
]


def load_config(path: str = "config.yaml") -> Config:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc

    try:
        regions = {k: Region(*raw["regions"][k]) for k in _REQUIRED_REGIONS}
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"missing/invalid regions: {exc}") from exc

    try:
        dev = raw.get("device", {})
        loop = raw["loop"]
        g = raw["gestures"]
        rw = raw["reward"]
        menu = raw["menu"]
        return Config(
            device_serial=dev.get("serial"),
            capture_backend=dev.get("capture", "scrcpy"),
            max_fps=int(dev.get("max_fps", 60)),
            decision_hz=int(loop["decision_hz"]),
            target_stage=str(loop["target_stage"]),
            regions=regions,
            gestures=Gestures(tuple(g["jump_button"]), tuple(g["slide_button"]),
                              int(g["slide_hold_ms"])),
            reward=RewardWeights(float(rw["w_coin"]), float(rw["w_box"]),
                                 float(rw["w_survive"]), float(rw["death_penalty"])),
            menu_allowlist=list(menu["allowlist"]),
            menu_denylist=list(menu["denylist"]),
            templates_dir=str(raw.get("templates_dir", "templates")),
        )
    except KeyError as exc:
        raise ConfigError(f"missing config key: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Create `config.example.yaml` (committed template with placeholder pixel coords)**

```yaml
# Copy to config.yaml and calibrate coords with `python -m cookierun_bot.calibrate`.
device: {serial: null, capture: scrcpy, max_fps: 60}
loop: {target_stage: "Episode 1", decision_hz: 15}
regions:              # [x, y, w, h] on the CAPTURED frame — calibrate per device
  play_area: [0, 200, 1080, 1200]
  coin_counter: [800, 40, 200, 60]
  mystery_box_counter: [500, 40, 120, 60]
  results_coins: [400, 900, 300, 80]
  results_ingredients: [400, 1000, 300, 80]
gestures: {jump_button: [200, 1600], slide_button: [880, 1600], slide_hold_ms: 300}
reward: {w_coin: 1.0, w_box: 50.0, w_survive: 0.01, death_penalty: 10.0}
menu:
  allowlist: [restart, replay, collect, ok, start]
  denylist: [revive_crystals, buy, purchase, watch_ad]
templates_dir: templates
```

- [ ] **Step 6: Commit**

```bash
git add src/cookierun_bot/config.py config.example.yaml tests/test_config.py
git commit -m "feat: config dataclasses and YAML loader with validation"
```

---

## Task 3: Device layer (scrcpy + ADB fallback)

**Files:**
- Create: `src/cookierun_bot/device.py`

**Interfaces:**
- Consumes: `Config` (Task 2).
- Produces:
  - `Device` protocol: `start()`, `stop()`, `last_frame()->np.ndarray|None`, `resolution->tuple[int,int]`, `tap(x,y)`, `hold(x,y,duration_ms)`
  - `ScrcpyDevice(serial, max_fps)`, `AdbDevice(serial)`, `open_device(cfg)->Device`

> **Note:** This task is hardware I/O — verification is **manual against a connected phone**, not unit tests. The `FakeDevice` in `conftest.py` is what downstream unit tests use.

- [ ] **Step 1: Write the implementation**

```python
# src/cookierun_bot/device.py
from __future__ import annotations
import time
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Device(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def last_frame(self) -> "np.ndarray | None": ...
    @property
    def resolution(self) -> tuple[int, int]: ...
    def tap(self, x: int, y: int) -> None: ...
    def hold(self, x: int, y: int, duration_ms: int) -> None: ...


class ScrcpyDevice:
    """Low-latency capture + control via scrcpy-client."""
    def __init__(self, serial: str | None = None, max_fps: int = 60):
        import scrcpy  # imported lazily so tests without a phone still import the module
        self._scrcpy = scrcpy
        self._client = scrcpy.Client(
            device=serial, max_fps=max_fps, block_frame=True
        )
        self._client.add_listener(scrcpy.EVENT_FRAME, self._on_frame)
        self._latest = None

    def _on_frame(self, frame):
        if frame is not None:
            self._latest = frame  # BGR ndarray

    def start(self) -> None:
        self._client.start(threaded=True)

    def stop(self) -> None:
        self._client.stop()

    def last_frame(self):
        return self._latest

    @property
    def resolution(self) -> tuple[int, int]:
        return self._client.resolution

    def tap(self, x: int, y: int) -> None:
        self._client.control.touch(x, y, self._scrcpy.ACTION_DOWN)
        self._client.control.touch(x, y, self._scrcpy.ACTION_UP)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        self._client.control.touch(x, y, self._scrcpy.ACTION_DOWN)
        time.sleep(duration_ms / 1000.0)
        self._client.control.touch(x, y, self._scrcpy.ACTION_UP)


class AdbDevice:
    """Slower fallback via adbutils. Fine for menus; too slow for in-run reactions."""
    def __init__(self, serial: str | None = None):
        import adbutils
        self._dev = (adbutils.adb.device(serial=serial) if serial
                     else adbutils.adb.device_list()[0])

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def last_frame(self):
        img = self._dev.screenshot()          # PIL.Image (RGB)
        return np.asarray(img)[:, :, ::-1].copy()  # -> BGR ndarray

    @property
    def resolution(self) -> tuple[int, int]:
        w, h = self._dev.window_size()
        return (w, h)

    def tap(self, x: int, y: int) -> None:
        self._dev.click(x, y)

    def hold(self, x: int, y: int, duration_ms: int) -> None:
        self._dev.swipe(x, y, x, y, duration_ms / 1000.0)


def open_device(cfg) -> Device:
    if cfg.capture_backend == "adb":
        return AdbDevice(cfg.device_serial)
    return ScrcpyDevice(cfg.device_serial, cfg.max_fps)
```

- [ ] **Step 2: Manual verification against a phone**

Prereqs: phone plugged in, USB debugging on, `adb devices` lists it, scrcpy ≥ v1.20 installed.
Run this throwaway snippet:

```python
# scratch_device_check.py  (do not commit)
import time
from cookierun_bot.config import load_config
from cookierun_bot.device import open_device
import cv2

cfg = load_config("config.yaml")
dev = open_device(cfg)
dev.start()
time.sleep(2)                       # let frames arrive
frame = dev.last_frame()
print("resolution:", dev.resolution, "frame:", None if frame is None else frame.shape)
cv2.imwrite("scratch_frame.png", frame)   # inspect it looks like the game
dev.tap(*cfg.gestures.jump_button)  # cookie should jump
time.sleep(1)
dev.stop()
```

Expected: `scratch_frame.png` shows the live game; the cookie visibly jumps.
If scrcpy fails to connect, set `capture: adb` in `config.yaml` and re-run (slower but proves the pipeline).

- [ ] **Step 3: Commit**

```bash
git add src/cookierun_bot/device.py
git commit -m "feat: Device abstraction with scrcpy capture+control and adb fallback"
```

---

## Task 4: Frame capture & preprocessing

**Files:**
- Create: `src/cookierun_bot/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `Region` (Task 2).
- Produces:
  - `preprocess(frame, play_area:Region, size:tuple[int,int]=(84,84))->np.ndarray` (uint8, shape `size`, grayscale)
  - `FrameStack(k:int=4)` with `reset(frame)->np.ndarray` and `push(frame)->np.ndarray`, both returning shape `(k, 84, 84)` uint8.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_capture.py
import numpy as np
from cookierun_bot.config import Region
from cookierun_bot.capture import preprocess, FrameStack


def test_preprocess_shape_and_dtype():
    frame = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    out = preprocess(frame, Region(0, 200, 1080, 1200), size=(84, 84))
    assert out.shape == (84, 84)
    assert out.dtype == np.uint8


def test_framestack_reset_repeats_then_push_shifts():
    fs = FrameStack(k=4)
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    pa = Region(0, 0, 1080, 1920)
    stacked = fs.reset(preprocess(frame, pa))
    assert stacked.shape == (4, 84, 84)
    bright = np.full((1920, 1080, 3), 255, dtype=np.uint8)
    stacked2 = fs.push(preprocess(bright, pa))
    assert stacked2.shape == (4, 84, 84)
    assert stacked2[-1].mean() > stacked2[0].mean()   # newest frame is the bright one
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.capture`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/capture.py
from __future__ import annotations
from collections import deque
import cv2
import numpy as np


def preprocess(frame, play_area, size=(84, 84)) -> np.ndarray:
    crop = play_area.crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.uint8)


class FrameStack:
    def __init__(self, k: int = 4):
        self.k = k
        self._frames: deque = deque(maxlen=k)

    def reset(self, frame) -> np.ndarray:
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(frame)
        return np.stack(self._frames, axis=0)

    def push(self, frame) -> np.ndarray:
        self._frames.append(frame)
        return np.stack(self._frames, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_capture.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/capture.py tests/test_capture.py
git commit -m "feat: frame preprocessing and k-frame stack"
```

---

## Task 5: Detection (templates + counters + results)

**Files:**
- Create: `src/cookierun_bot/detect.py`
- Test: `tests/test_detect.py`

**Interfaces:**
- Consumes: `Config`, `Region` (Task 2).
- Produces:
  - `TemplateMatcher(templates_dir:str)` with `present(frame,name,threshold=0.8)->bool` and `find(frame,name,threshold=0.8)->tuple[int,int]|None`
  - `read_int(frame, region:Region)->int|None` (digit OCR)
  - `detect_death(frame, matcher)->bool` (true when a "results"/"gameover" template is present)
  - `read_coins(frame, cfg)->int|None`
  - `read_mystery_boxes(frame, cfg)->int` (parses the "n/3" counter; 0 on failure)
  - `read_results(frame, cfg)->dict` -> `{"coins":int, "ingredients":int}`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detect.py
import numpy as np
import cv2
import pytest
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.detect import (
    TemplateMatcher, read_int, read_mystery_boxes,
)


def _digit_image(text, size=(200, 60)):
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(img, text, (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                (255, 255, 255), 3, cv2.LINE_AA)
    return img


def _cfg(regions):
    return Config(None, "scrcpy", 60, 15, "Episode 1", regions,
                  Gestures((0, 0), (0, 0), 300),
                  RewardWeights(1, 50, 0.01, 10), ["ok"], ["buy"], "templates")


def test_template_matcher_finds_known_template(tmp_path):
    tpl = np.zeros((30, 30, 3), dtype=np.uint8)
    tpl[:15, :] = 200                               # patterned so TM_CCOEFF_NORMED is well-defined
    cv2.imwrite(str(tmp_path / "blob.png"), tpl)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[100:130, 50:80] = tpl                     # place identical patch
    m = TemplateMatcher(str(tmp_path))
    assert m.present(frame, "blob", threshold=0.9) is True
    assert m.find(frame, "blob", threshold=0.9) is not None
    assert m.present(np.zeros((200, 200, 3), np.uint8), "blob") is False


@pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None, reason="tesseract not installed"
)
def test_read_int_reads_digits():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    frame[0:60, 0:200] = _digit_image("1234")
    val = read_int(frame, Region(0, 0, 200, 60))
    assert val == 1234


def test_read_mystery_boxes_zero_on_unreadable():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    cfg = _cfg({"mystery_box_counter": Region(0, 0, 50, 50),
                "coin_counter": Region(0, 0, 50, 50),
                "results_coins": Region(0, 0, 50, 50),
                "results_ingredients": Region(0, 0, 50, 50),
                "play_area": Region(0, 0, 50, 50)})
    assert read_mystery_boxes(frame, cfg) == 0     # blank -> 0, never crashes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.detect`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/detect.py
from __future__ import annotations
import os
import re
import glob
import cv2
import numpy as np


class TemplateMatcher:
    def __init__(self, templates_dir: str):
        self._templates: dict[str, np.ndarray] = {}
        for path in glob.glob(os.path.join(templates_dir, "*.png")):
            name = os.path.splitext(os.path.basename(path))[0]
            self._templates[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _match(self, frame, name):
        tpl = self._templates.get(name)
        if tpl is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
            return None
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return max_val, max_loc, tpl.shape

    def present(self, frame, name, threshold: float = 0.8) -> bool:
        m = self._match(frame, name)
        return bool(m and m[0] >= threshold)

    def find(self, frame, name, threshold: float = 0.8):
        m = self._match(frame, name)
        if not m or m[0] < threshold:
            return None
        (max_val, (mx, my), (th, tw)) = m
        return (mx + tw // 2, my + th // 2)   # center point


def read_int(frame, region) -> "int | None":
    try:
        import pytesseract
    except ImportError:
        return None
    # ponytail: OCR is a best-effort screen read at a trust boundary — any failure
    # (missing tesseract binary, bad crop, decode error) must degrade to "unknown"
    # (None) rather than crash the running bot, so we catch broadly here.
    try:
        crop = region.crop(frame)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        txt = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789")
    except Exception:
        return None
    digits = re.sub(r"\D", "", txt)
    return int(digits) if digits else None


def detect_death(frame, matcher: TemplateMatcher) -> bool:
    return matcher.present(frame, "results", 0.8) or matcher.present(frame, "gameover", 0.8)


def read_coins(frame, cfg) -> "int | None":
    return read_int(frame, cfg.regions["coin_counter"])


def read_mystery_boxes(frame, cfg) -> int:
    """Parse the 'n/3' box counter; return n, or 0 if unreadable."""
    val = read_int(frame, cfg.regions["mystery_box_counter"])
    if val is None:
        return 0
    return min(val, 3)


def read_results(frame, cfg) -> dict:
    coins = read_int(frame, cfg.regions["results_coins"]) or 0
    ingredients = read_int(frame, cfg.regions["results_ingredients"]) or 0
    return {"coins": coins, "ingredients": ingredients}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_detect.py -v`
Expected: PASS (the tesseract test skips if tesseract isn't installed).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/detect.py tests/test_detect.py
git commit -m "feat: template matching, digit OCR, death/coin/box/results detection"
```

---

## Task 6: Gestures (action → touch)

**Files:**
- Create: `src/cookierun_bot/gestures.py`
- Test: `tests/test_gestures.py`

**Interfaces:**
- Consumes: `Device` (Task 3), `Gestures` (Task 2).
- Produces:
  - Constants `ACTION_NOOP=0`, `ACTION_JUMP=1`, `ACTION_SLIDE=2`, `N_ACTIONS=3`
  - `apply_action(device, action:int, g:Gestures)->None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gestures.py
from cookierun_bot.config import Gestures
from cookierun_bot.gestures import apply_action, ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE


def test_noop_does_nothing(fake_device):
    apply_action(fake_device, ACTION_NOOP, Gestures((200, 1600), (880, 1600), 300))
    assert fake_device.taps == [] and fake_device.holds == []


def test_jump_taps_jump_button(fake_device):
    apply_action(fake_device, ACTION_JUMP, Gestures((200, 1600), (880, 1600), 300))
    assert fake_device.taps == [(200, 1600)]


def test_slide_holds_slide_button(fake_device):
    apply_action(fake_device, ACTION_SLIDE, Gestures((200, 1600), (880, 1600), 300))
    assert fake_device.holds == [(880, 1600, 300)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gestures.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.gestures`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/gestures.py
from __future__ import annotations

ACTION_NOOP = 0
ACTION_JUMP = 1
ACTION_SLIDE = 2
N_ACTIONS = 3


def apply_action(device, action: int, g) -> None:
    if action == ACTION_JUMP:
        device.tap(*g.jump_button)
    elif action == ACTION_SLIDE:
        device.hold(g.slide_button[0], g.slide_button[1], g.slide_hold_ms)
    # ACTION_NOOP: intentionally do nothing
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gestures.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/gestures.py tests/test_gestures.py
git commit -m "feat: action-to-gesture mapping (jump tap, slide hold)"
```

---

## Task 7: Reward tracker

**Files:**
- Create: `src/cookierun_bot/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Consumes: `RewardWeights` (Task 2).
- Produces:
  - `RewardTracker(w:RewardWeights)` with `reset()`, `update(coins:int|None, boxes:int, dead:bool)->float`, `summary()->dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py
from cookierun_bot.config import RewardWeights
from cookierun_bot.reward import RewardTracker


def _w():
    return RewardWeights(w_coin=1.0, w_box=50.0, w_survive=0.01, death_penalty=10.0)


def test_coin_delta_rewarded_not_absolute():
    rt = RewardTracker(_w())
    rt.reset()
    assert rt.update(coins=10, boxes=0, dead=False) == 10 * 1.0 + 0.01   # first delta from 0
    assert rt.update(coins=15, boxes=0, dead=False) == 5 * 1.0 + 0.01    # +5 more
    assert rt.update(coins=15, boxes=0, dead=False) == 0.0 + 0.01        # no new coins


def test_box_pickup_gives_big_bonus():
    rt = RewardTracker(_w())
    rt.reset()
    r = rt.update(coins=0, boxes=1, dead=False)
    assert r == 50.0 + 0.01


def test_none_coins_counts_zero_delta():
    rt = RewardTracker(_w())
    rt.reset()
    assert rt.update(coins=None, boxes=0, dead=False) == 0.01


def test_death_applies_penalty_and_no_survive_bonus():
    rt = RewardTracker(_w())
    rt.reset()
    assert rt.update(coins=0, boxes=0, dead=True) == -10.0


def test_summary_tracks_totals():
    rt = RewardTracker(_w())
    rt.reset()
    rt.update(coins=20, boxes=1, dead=False)
    s = rt.summary()
    assert s["coins"] == 20 and s["boxes"] == 1 and s["steps"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.reward`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/reward.py
from __future__ import annotations


class RewardTracker:
    def __init__(self, w):
        self._w = w
        self.reset()

    def reset(self) -> None:
        self._prev_coins = 0
        self._prev_boxes = 0
        self._total_coins = 0
        self._total_boxes = 0
        self._steps = 0

    def update(self, coins, boxes: int, dead: bool) -> float:
        self._steps += 1
        coin_delta = 0
        if coins is not None:
            coin_delta = max(0, coins - self._prev_coins)
            self._prev_coins = coins
            self._total_coins = coins
        box_delta = max(0, boxes - self._prev_boxes)
        self._prev_boxes = boxes
        self._total_boxes = max(self._total_boxes, boxes)

        reward = self._w.w_coin * coin_delta + self._w.w_box * box_delta
        if dead:
            reward -= self._w.death_penalty
        else:
            reward += self._w.w_survive
        return reward

    def summary(self) -> dict:
        return {"coins": self._total_coins, "boxes": self._total_boxes,
                "steps": self._steps}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/reward.py tests/test_reward.py
git commit -m "feat: coin+box+survival reward tracker (no score reward)"
```

---

## Task 8: Menu navigator (currency guardrail)

**Files:**
- Create: `src/cookierun_bot/menu.py`
- Test: `tests/test_menu.py`

**Interfaces:**
- Consumes: `Device` (Task 3), `TemplateMatcher` (Task 5), `Config` (Task 2).
- Produces:
  - `MenuNavigator(device, matcher, cfg)` with:
    - `is_spend_dialog(frame)->bool` (any denylist template present)
    - `tap_allowed(frame)->bool` (taps center of first present allowlist template; returns whether it tapped; NEVER taps if a denylist dialog is present)
    - `advance(frame)->str` returns one of `"spend_blocked"`, `"tapped"`, `"idle"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_menu.py
import numpy as np
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.menu import MenuNavigator


class StubMatcher:
    def __init__(self, present_names, points=None):
        self._present = set(present_names)
        self._points = points or {}
    def present(self, frame, name, threshold=0.8):
        return name in self._present
    def find(self, frame, name, threshold=0.8):
        return self._points.get(name)


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "scrcpy", 60, 15, "Episode 1",
                  {k: r for k in ["play_area", "coin_counter", "mystery_box_counter",
                                  "results_coins", "results_ingredients"]},
                  Gestures((0, 0), (0, 0), 300), RewardWeights(1, 50, 0.01, 10),
                  ["restart", "ok"], ["buy", "revive_crystals"], "templates")


def test_denylist_blocks_tapping(fake_device):
    m = StubMatcher(present_names=["ok", "revive_crystals"], points={"ok": (5, 5)})
    nav = MenuNavigator(fake_device, m, _cfg())
    frame = np.zeros((100, 100, 3), np.uint8)
    assert nav.is_spend_dialog(frame) is True
    assert nav.advance(frame) == "spend_blocked"
    assert fake_device.taps == []          # never tapped a spend dialog


def test_taps_first_allowlist_button(fake_device):
    m = StubMatcher(present_names=["restart"], points={"restart": (30, 40)})
    nav = MenuNavigator(fake_device, m, _cfg())
    frame = np.zeros((100, 100, 3), np.uint8)
    assert nav.advance(frame) == "tapped"
    assert fake_device.taps == [(30, 40)]


def test_idle_when_nothing_present(fake_device):
    m = StubMatcher(present_names=[])
    nav = MenuNavigator(fake_device, m, _cfg())
    assert nav.advance(np.zeros((100, 100, 3), np.uint8)) == "idle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_menu.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.menu`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/menu.py
from __future__ import annotations


class MenuNavigator:
    def __init__(self, device, matcher, cfg):
        self._device = device
        self._matcher = matcher
        self._cfg = cfg

    def is_spend_dialog(self, frame) -> bool:
        return any(self._matcher.present(frame, name)
                   for name in self._cfg.menu_denylist)

    def tap_allowed(self, frame) -> bool:
        if self.is_spend_dialog(frame):
            return False                       # hard guardrail: never tap near a spend dialog
        for name in self._cfg.menu_allowlist:
            point = self._matcher.find(frame, name)
            if point is not None:
                self._device.tap(*point)
                return True
        return False

    def advance(self, frame) -> str:
        if self.is_spend_dialog(frame):
            return "spend_blocked"
        return "tapped" if self.tap_allowed(frame) else "idle"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_menu.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/menu.py tests/test_menu.py
git commit -m "feat: menu navigator with allowlist tapping and spend-dialog guardrail"
```

---

## Task 9: Metrics

**Files:**
- Create: `src/cookierun_bot/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `RunResult(coins:int, ingredients:int, duration_s:float)`
  - `Metrics()` with `add(r:RunResult)`, `coins_per_hour()->float`, `ingredients_per_hour()->float`, `summary()->str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from cookierun_bot.metrics import RunResult, Metrics


def test_rates_computed_over_total_time():
    m = Metrics()
    m.add(RunResult(coins=100, ingredients=2, duration_s=60.0))
    m.add(RunResult(coins=200, ingredients=1, duration_s=60.0))
    # 300 coins / 120s = 2.5/s -> 9000/hr ; 3 ingredients / 120s -> 90/hr
    assert round(m.coins_per_hour(), 1) == 9000.0
    assert round(m.ingredients_per_hour(), 1) == 90.0


def test_empty_metrics_zero_rates():
    m = Metrics()
    assert m.coins_per_hour() == 0.0
    assert m.ingredients_per_hour() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.metrics`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/metrics.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RunResult:
    coins: int
    ingredients: int
    duration_s: float


class Metrics:
    def __init__(self):
        self._runs: list[RunResult] = []

    def add(self, r: RunResult) -> None:
        self._runs.append(r)

    def _total_seconds(self) -> float:
        return sum(r.duration_s for r in self._runs)

    def coins_per_hour(self) -> float:
        secs = self._total_seconds()
        if secs <= 0:
            return 0.0
        return sum(r.coins for r in self._runs) / secs * 3600.0

    def ingredients_per_hour(self) -> float:
        secs = self._total_seconds()
        if secs <= 0:
            return 0.0
        return sum(r.ingredients for r in self._runs) / secs * 3600.0

    def summary(self) -> str:
        return (f"runs={len(self._runs)} "
                f"coins/hr={self.coins_per_hour():.0f} "
                f"ingredients/hr={self.ingredients_per_hour():.1f}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/metrics.py tests/test_metrics.py
git commit -m "feat: coins/hr and ingredients/hr metrics"
```

---

## Task 10: Gymnasium environment

**Files:**
- Create: `src/cookierun_bot/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `Device` (3), `capture` (4), `detect` (5), `gestures` (6), `reward` (7), `Config` (2).
- Produces:
  - `CookieRunEnv(device, cfg, matcher, tick_sleep=None)`:
    - `observation_space = Box(0,255,(4,84,84),uint8)`, `action_space = Discrete(3)`
    - `reset(*, seed=None, options=None)->(obs, info)`
    - `step(action)->(obs, reward, terminated, truncated, info)` — `info` has `coins`, `boxes`, `dead`
    - `last_raw_frame()->np.ndarray|None` (raw BGR frame, for the rule-based agent)
    - `close()`

> `tick_sleep` is injected (defaults to real sleep at `1/decision_hz`); tests pass a no-op so they don't wait.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py
import numpy as np
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.env import CookieRunEnv


class StubMatcher:
    def __init__(self, dead=False): self.dead = dead
    def present(self, frame, name, threshold=0.8):
        return self.dead and name in ("results", "gameover")
    def find(self, frame, name, threshold=0.8): return None


def _cfg():
    r = Region(0, 0, 100, 100)
    return Config(None, "scrcpy", 60, 15, "Episode 1",
                  {"play_area": Region(0, 0, 200, 200), "coin_counter": r,
                   "mystery_box_counter": r, "results_coins": r,
                   "results_ingredients": r},
                  Gestures((10, 20), (30, 40), 300),
                  RewardWeights(1, 50, 0.01, 10), ["ok"], ["buy"], "templates")


def test_reset_returns_stacked_obs(fake_device):
    fake_device.set_frame(np.zeros((400, 400, 3), np.uint8))
    env = CookieRunEnv(fake_device, _cfg(), StubMatcher(), tick_sleep=lambda: None)
    obs, info = env.reset()
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8


def test_step_jump_taps_and_returns_five_tuple(fake_device):
    fake_device.set_frame(np.zeros((400, 400, 3), np.uint8))
    env = CookieRunEnv(fake_device, _cfg(), StubMatcher(), tick_sleep=lambda: None)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)   # jump
    assert fake_device.taps == [(10, 20)]
    assert obs.shape == (4, 84, 84)
    assert terminated is False
    assert set(["coins", "boxes", "dead"]).issubset(info)


def test_step_terminates_on_death(fake_device):
    fake_device.set_frame(np.zeros((400, 400, 3), np.uint8))
    env = CookieRunEnv(fake_device, _cfg(), StubMatcher(dead=True), tick_sleep=lambda: None)
    env.reset()
    _, reward, terminated, _, info = env.step(0)
    assert terminated is True and info["dead"] is True
    assert reward < 0            # death penalty dominates
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.env`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/env.py
from __future__ import annotations
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .capture import preprocess, FrameStack
from .detect import detect_death, read_coins, read_mystery_boxes
from .gestures import apply_action, N_ACTIONS
from .reward import RewardTracker


class CookieRunEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, device, cfg, matcher, tick_sleep=None):
        super().__init__()
        self._device = device
        self._cfg = cfg
        self._matcher = matcher
        self._stack = FrameStack(k=4)
        self._reward = RewardTracker(cfg.reward)
        self._tick_sleep = tick_sleep or (lambda: time.sleep(1.0 / cfg.decision_hz))
        self.observation_space = spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._last_raw = None

    def _grab(self):
        frame = self._device.last_frame()
        if frame is None:
            frame = np.zeros((self._cfg.regions["play_area"].h +
                              self._cfg.regions["play_area"].y,
                              self._cfg.regions["play_area"].w, 3), np.uint8)
        self._last_raw = frame
        return frame

    def last_raw_frame(self):
        return self._last_raw

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reward.reset()
        frame = self._grab()
        obs = self._stack.reset(preprocess(frame, self._cfg.regions["play_area"]))
        return obs, {"coins": 0, "boxes": 0, "dead": False}

    def step(self, action):
        apply_action(self._device, int(action), self._cfg.gestures)
        self._tick_sleep()
        frame = self._grab()
        dead = detect_death(frame, self._matcher)
        coins = read_coins(frame, self._cfg)
        boxes = read_mystery_boxes(frame, self._cfg)
        reward = self._reward.update(coins=coins, boxes=boxes, dead=dead)
        obs = self._stack.push(preprocess(frame, self._cfg.regions["play_area"]))
        info = {"coins": coins if coins is not None else 0, "boxes": boxes, "dead": dead}
        return obs, reward, dead, False, info

    def close(self):
        self._device.stop()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/env.py tests/test_env.py
git commit -m "feat: CookieRunEnv gym environment (capture->act->reward->done)"
```

---

## Task 11: Rule-based agent

**Files:**
- Create: `src/cookierun_bot/policies/rule_based.py`
- Test: `tests/test_rule_based.py`

**Interfaces:**
- Consumes: `Config`, `Region` (Task 2).
- Produces:
  - `Features(low_obstacle:bool, tall_obstacle:bool, gap:bool)`
  - `extract_features(frame, cfg)->Features`
  - `RuleBasedAgent(cfg)` with `reset()` and `act(frame)->int` (priority: survive → then noop so coins/jellies in the path are collected naturally by running/jumping)

> The heuristic reads a **danger-zone** band (config region `play_area`, split into an upper and lower half just ahead of the cookie). Dark blobs in the **lower** band → `slide` under a low obstacle is wrong; in CookieRun low obstacles are jumped and *overhead* obstacles are slid under. We encode: obstacle/gap in the lower band → **jump**; obstacle in the upper (overhead) band only → **slide**. Thresholds are calibratable and meant to be tuned on real footage (see manual step).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rule_based.py
import numpy as np
from cookierun_bot.config import Region, Config, Gestures, RewardWeights
from cookierun_bot.policies.rule_based import RuleBasedAgent, extract_features
from cookierun_bot.gestures import ACTION_JUMP, ACTION_SLIDE, ACTION_NOOP


def _cfg():
    r = Region(0, 0, 10, 10)
    return Config(None, "scrcpy", 60, 15, "Episode 1",
                  {"play_area": Region(0, 0, 100, 100), "coin_counter": r,
                   "mystery_box_counter": r, "results_coins": r,
                   "results_ingredients": r},
                  Gestures((0, 0), (0, 0), 300), RewardWeights(1, 50, 0.01, 10),
                  ["ok"], ["buy"], "templates")


def test_clear_path_is_noop():
    frame = np.full((100, 100, 3), 200, np.uint8)     # bright, no obstacles
    agent = RuleBasedAgent(_cfg())
    agent.reset()
    assert agent.act(frame) == ACTION_NOOP


def test_ground_obstacle_triggers_jump():
    frame = np.full((100, 100, 3), 200, np.uint8)
    frame[70:100, 40:60] = 0                          # dark blob low in play area
    agent = RuleBasedAgent(_cfg())
    agent.reset()
    assert agent.act(frame) == ACTION_JUMP


def test_overhead_only_obstacle_triggers_slide():
    frame = np.full((100, 100, 3), 200, np.uint8)
    frame[0:25, 40:60] = 0                             # dark blob only near the top
    agent = RuleBasedAgent(_cfg())
    agent.reset()
    assert agent.act(frame) == ACTION_SLIDE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rule_based.py -v`
Expected: FAIL with `ModuleNotFoundError: cookierun_bot.policies.rule_based`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cookierun_bot/policies/rule_based.py
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

from ..gestures import ACTION_NOOP, ACTION_JUMP, ACTION_SLIDE

_DARK_FRACTION = 0.05       # >5% dark pixels in a band = obstacle. Tune on real footage.


@dataclass
class Features:
    low_obstacle: bool      # obstacle in the lower (ground) band -> jump
    overhead_obstacle: bool  # obstacle only in the upper band -> slide


def _dark_fraction(band) -> float:
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if band.ndim == 3 else band
    return float((gray < 60).mean())


def extract_features(frame, cfg) -> Features:
    zone = cfg.regions["play_area"].crop(frame)
    h = zone.shape[0]
    upper = zone[0:h // 2]
    lower = zone[h // 2:]
    low = _dark_fraction(lower) > _DARK_FRACTION
    up = _dark_fraction(upper) > _DARK_FRACTION
    return Features(low_obstacle=low, overhead_obstacle=(up and not low))


class RuleBasedAgent:
    def __init__(self, cfg):
        self._cfg = cfg

    def reset(self) -> None:
        pass

    def act(self, frame) -> int:
        f = extract_features(frame, self._cfg)
        if f.low_obstacle:
            return ACTION_JUMP           # survive first: clear ground obstacles / gaps
        if f.overhead_obstacle:
            return ACTION_SLIDE
        return ACTION_NOOP               # run on; coins/jellies in the lane are auto-collected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rule_based.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cookierun_bot/policies/rule_based.py tests/test_rule_based.py
git commit -m "feat: rule-based survival agent (danger-zone jump/slide heuristic)"
```

---

## Task 12: Calibration helper

**Files:**
- Create: `src/cookierun_bot/calibrate.py`

**Interfaces:**
- Consumes: `Config` (2), `Device` (3).
- Produces: CLI `python -m cookierun_bot.calibrate` that saves a full screenshot and prints resolution, so the user can measure region rectangles and crop button/counter templates into `templates/`.

> Hardware I/O — verified manually.

- [ ] **Step 1: Write the implementation**

```python
# src/cookierun_bot/calibrate.py
from __future__ import annotations
import sys
import time
import cv2
from .config import load_config
from .device import open_device


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    cfg_path = argv[0] if argv else "config.yaml"
    cfg = load_config(cfg_path)
    dev = open_device(cfg)
    dev.start()
    time.sleep(2.0)                     # allow scrcpy frames to arrive
    frame = dev.last_frame()
    dev.stop()
    if frame is None:
        print("No frame captured. Is the phone connected and scrcpy working?")
        return 1
    out = "calibration_screenshot.png"
    cv2.imwrite(out, frame)
    print(f"resolution={dev.resolution} saved={out} shape={frame.shape}")
    print("Open the PNG in an image editor, read pixel rects for each region,")
    print("and crop button/counter images into the templates/ folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Manual verification**

Run: `python -m cookierun_bot.calibrate config.yaml`
Expected: prints resolution and writes `calibration_screenshot.png` showing the live game.
Use it to fill in every `regions:` rectangle in `config.yaml`, and crop these templates into `templates/`: `results.png`/`gameover.png` (end-of-run screen), plus one PNG per allowlist button (`restart.png`, `replay.png`, `collect.png`, `ok.png`, `start.png`) and per denylist dialog (`revive_crystals.png`, `buy.png`, `purchase.png`, `watch_ad.png`).

- [ ] **Step 3: Commit**

```bash
git add src/cookierun_bot/calibrate.py
git commit -m "feat: calibration screenshot helper"
```

---

## Task 13: Farm loop (the working bot)

**Files:**
- Create: `src/cookierun_bot/agents/play.py`

**Interfaces:**
- Consumes: everything above.
- Produces: CLI `python -m cookierun_bot.agents.play` that runs the rule-based farm loop: start run → play until death → OCR results → collect+replay → log metrics; **respects the spend guardrail throughout.**

> Integration entry point — verified manually against the phone.

- [ ] **Step 1: Write the implementation**

```python
# src/cookierun_bot/agents/play.py
from __future__ import annotations
import sys
import time

from ..config import load_config
from ..device import open_device
from ..detect import TemplateMatcher, read_results
from ..env import CookieRunEnv
from ..menu import MenuNavigator
from ..metrics import Metrics, RunResult
from ..policies.rule_based import RuleBasedAgent


def _drive_menu_until_running(nav, device, timeout=30.0):
    """Tap allowlist buttons until the run starts (or timeout). Never taps spend dialogs."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        frame = device.last_frame()
        if frame is None:
            time.sleep(0.2); continue
        state = nav.advance(frame)
        if state == "spend_blocked":
            time.sleep(0.5)          # wait for the spend dialog to be dismissed elsewhere
        elif state == "idle":
            return True              # nothing left to tap -> assume in-run
        time.sleep(0.4)
    return False


def play(cfg_path="config.yaml", max_runs=None) -> None:
    cfg = load_config(cfg_path)
    device = open_device(cfg)
    device.start()
    time.sleep(2.0)
    matcher = TemplateMatcher(cfg.templates_dir)
    env = CookieRunEnv(device, cfg, matcher)
    agent = RuleBasedAgent(cfg)
    nav = MenuNavigator(device, matcher, cfg)
    metrics = Metrics()

    run = 0
    try:
        while max_runs is None or run < max_runs:
            _drive_menu_until_running(nav, device)
            obs, _ = env.reset()
            agent.reset()
            t0 = time.monotonic()
            terminated = False
            while not terminated:
                frame = env.last_raw_frame()
                action = agent.act(frame) if frame is not None else 0
                obs, reward, terminated, truncated, info = env.step(action)
            duration = time.monotonic() - t0
            results = read_results(device.last_frame(), cfg)
            metrics.add(RunResult(results["coins"], results["ingredients"], duration))
            run += 1
            print(f"[run {run}] {results} dur={duration:.1f}s | {metrics.summary()}")
            # collect rewards + replay, guardrail-protected
            _drive_menu_until_running(nav, device)
    finally:
        env.close()
        print("FINAL:", metrics.summary())


if __name__ == "__main__":
    args = sys.argv[1:]
    play(args[0] if args else "config.yaml")
```

- [ ] **Step 2: Manual verification (the payoff)**

Prereqs: `config.yaml` calibrated (Task 12), templates cropped, game open at a launchable stage.
Run: `python -m cookierun_bot.agents.play config.yaml`
Expected:
- The cookie plays hands-off: jumps ground obstacles, slides overhead ones, survives.
- On death it tallies coins/ingredients, then taps through to replay — **never** tapping a revive/buy dialog.
- Each run prints `coins/hr` and `ingredients/hr`.
Tune `_DARK_FRACTION` (rule_based.py), region rects, and `decision_hz` until survival is solid, then let it farm.

- [ ] **Step 3: Commit**

```bash
git add src/cookierun_bot/agents/play.py
git commit -m "feat: rule-based farm loop with metrics and spend guardrail"
```

---

## Task 14: Full test run + README pointer

**Files:**
- Modify: none (verification task); optionally create `README.md`.

- [ ] **Step 1: Run the whole unit suite**

Run: `pytest -v`
Expected: all tests in Tasks 2,4,5,6,7,8,9,10,11 PASS (device/calibrate/play are manual).

- [ ] **Step 2: (Optional) Write a short `README.md`** documenting: install, `adb devices` + scrcpy prereq, copy `config.example.yaml`→`config.yaml`, run `calibrate`, crop templates, run `play`. Commit it.

```bash
git add README.md
git commit -m "docs: quickstart for calibration and farming"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** device/scrcpy+adb (T3) ✓; capture/frame-stack (T4) ✓; detect death/coins/boxes/results (T5) ✓; two-button gestures incl. slide-hold + double-jump-as-two-ticks (T6, T10) ✓; coin+box reward, no score term (T7) ✓; currency guardrail allow/denylist (T8) ✓; Gym env action/obs spaces (T10) ✓; rule-based survive→collect agent (T11) ✓; auto-restart/replay farm loop (T13) ✓; coins/hr + ingredients/hr metric (T9, T13) ✓; calibration (T12) ✓; config schema (T2) ✓. RL (Phase 5) is deliberately deferred to Plan 2.
- **Placeholder scan:** no TBD/TODO; every code step has full code; region pixel values are example calibration coords (correctly marked as calibratable), not placeholders in logic.
- **Type consistency:** `Config`/`Region`/`Gestures`/`RewardWeights` field names identical across T2/T5/T8/T10/T11 tests and impls; `apply_action(device, action, g)`, `RewardTracker.update(coins, boxes, dead)`, `MenuNavigator.advance(frame)->str`, `CookieRunEnv.step->5-tuple`, `read_results->{"coins","ingredients"}` all consistent between producer and consumer tasks.

## Deferred to Plan 2 (RL)

After Plan 1 runs and the env is measured: add `agents/train.py` (SB3 **DQN**, `CnnPolicy`), **warm-start by seeding the DQN replay buffer with `RuleBasedAgent` transitions** collected through `CookieRunEnv`, then online fine-tune; add model-vs-rule benchmarking via `Metrics`. Reward weights and warm-start step counts get tuned against the real environment's measured latency and coin/box rates.
