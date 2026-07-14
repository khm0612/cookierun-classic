from __future__ import annotations
from dataclasses import replace
import os
import queue
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from ..config import CAPTURE_BACKENDS, ConfigError, load_config
from ..detect import TemplateMatcher
from ..device import open_device, select_adb_serial
from ..farm import farm, format_boost_gate_status, read_boost_gate_status
from ..policies.rule_based import StreamingRuleBasedAgent
from .action_watch import format_action_sample, read_action_sample


def parse_max_runs(text: str) -> int | None:
    text = text.strip()
    if not text or text == "0":
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError("max runs must be a number") from exc
    if value <= 0:
        raise ValueError("max runs must be positive")
    return value


def parse_adb_devices(output: str) -> list[str]:
    devices = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def format_coin_line(latest: str, total: str, net: str) -> str:
    return (
        f"Latest coins: {format_count(latest)}    "
        f"Total: {format_count(total)}    Net: {format_count(net)}"
    )


def format_rate_line(net_rate: str, ingredients_rate: str) -> str:
    return f"Net/hr: {format_rate(net_rate)}    Ingredients/hr: {format_rate(ingredients_rate)}"


def format_status_line(status: str, runs: str, device: str) -> str:
    device = device.strip() or "no device"
    return f"Status: {status}    Runs: {runs}    Device: {device}"


def format_boost_line(text: str) -> str:
    text = text.strip() or "not checked"
    return f"Boost gate: {text}"


def format_action_line(text: str) -> str:
    text = text.strip() or "none"
    return f"Last action: {text}"


def format_action_brief(sample) -> str:
    return f"{sample.action_name} reason={sample.reason} confirmed={sample.confirmed}"


def summarize_action_log(line: str) -> str | None:
    if not line.startswith("[action] "):
        return None
    return line.removeprefix("[action] ").strip() or None


def summarize_boost_log(line: str) -> str | None:
    if not line.startswith("[boost] "):
        return None
    msg = line.removeprefix("[boost] ").strip()
    if msg.startswith("ready="):
        parts = dict(
            part.split("=", 1) for part in msg.split()
            if "=" in part and part.split("=", 1)[0] in {"ready", "required", "double_banner"}
        )
        ready = parts.get("ready", "?")
        required = parts.get("required", "?")
        double = parts.get("double_banner", "?")
        return f"ready={ready} required={required} double={double}"
    if "Double Coins" in msg or "required three" in msg:
        return msg
    return None


def format_count(value: str) -> str:
    text = str(value).strip()
    if not text or text == "-":
        return text
    try:
        return f"{int(text.replace(',', '')):,}"
    except ValueError:
        return text


def format_rate(value: str) -> str:
    text = str(value).strip()
    if not text.endswith("/hr"):
        return format_count(text)
    amount = text[:-3]
    return f"{format_count(amount)}/hr"


def discover_adb_path(preferred: str = "") -> str:
    preferred = preferred.strip()
    if preferred and os.path.exists(preferred):
        return preferred
    env_path = os.environ.get("ADBUTILS_ADB_PATH", "")
    if env_path and os.path.exists(env_path):
        return env_path
    found = shutil.which("adb")
    if found:
        return found
    try:
        import adbutils
        return adbutils.adb_path()
    except Exception:
        return preferred


def adb_ready_devices(adb: str) -> list[str]:
    out = subprocess.run(
        [adb or "adb", "devices"],
        capture_output=True,
        text=True,
        timeout=4,
        check=False,
    )
    if out.returncode != 0:
        return []
    return parse_adb_devices(out.stdout)


def controller_runtime_config(cfg, device_serial: str, capture_backend: str, adb_path: str):
    if capture_backend not in CAPTURE_BACKENDS:
        raise ConfigError(f"unknown capture backend: {capture_backend}")
    return replace(
        cfg,
        device_serial=device_serial.strip() or None,
        capture_backend=capture_backend,
        adb_path=adb_path.strip(),
    )


class ControllerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CookieGame")
        width = min(740, max(620, self.root.winfo_screenwidth() - 80))
        height = min(1030, max(860, self.root.winfo_screenheight() - 80))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(620, 760)
        self._events: queue.Queue[tuple] = queue.Queue()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None
        self._closing = False

        self.config_path = tk.StringVar(value="config.yaml")
        self.adb_path = tk.StringVar(value="")
        self.device_serial = tk.StringVar(value="")
        self.capture_backend = tk.StringVar(value="scrcpy")
        self.max_runs = tk.StringVar(value="0")
        self.allow_boosts = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="stopped")
        self.device_status = tk.StringVar(value="not checked")
        self.run_count = tk.StringVar(value="0")
        self.latest_coins = tk.StringVar(value="-")
        self.total_gross = tk.StringVar(value="0")
        self.total_net = tk.StringVar(value="0")
        self.net_rate = tk.StringVar(value="0/hr")
        self.ingredients_rate = tk.StringVar(value="0/hr")
        self.status_line = tk.StringVar(value=format_status_line("idle", "0", ""))
        self.coin_line = tk.StringVar(value=format_coin_line("-", "0", "0"))
        self.rate_line = tk.StringVar(value=format_rate_line("0/hr", "0/hr"))
        self.boost_line = tk.StringVar(value=format_boost_line(""))
        self.action_line = tk.StringVar(value=format_action_line(""))

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.load_defaults()
        self.root.after(100, self._poll_events)

    def _build(self) -> None:
        bg = "#242333"
        section_bg = "#343049"
        gold = "#ffd95c"
        self.root.configure(bg=bg)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=bg, foreground="#f5f0d0",
                        fieldbackground="#ffffff", font=("Segoe UI", 10))
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground="#f5f0d0")
        style.configure("Header.TLabel", background=bg, foreground=gold)
        style.configure("Muted.TLabel", background=bg, foreground="#7fc7e8")
        style.configure("Good.TLabel", background=bg, foreground="#58df9c")
        style.configure("Coin.TLabel", background=bg, foreground=gold)
        style.configure("Section.TFrame", background=section_bg)
        style.configure("Section.TLabel", background=section_bg, foreground="#f5f0d0")
        style.configure("SectionMuted.TLabel", background=section_bg, foreground="#7f8aa8")
        style.configure("Primary.TButton", padding=10, font=("Segoe UI", 12, "bold"))
        style.configure("Danger.TButton", padding=10, font=("Segoe UI", 12, "bold"),
                        background="#ec5353", foreground="#ffffff")
        style.map("Danger.TButton", background=[("active", "#ff6b6b")])

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(anchor="center", pady=(4, 12))
        logo = tk.Canvas(header, width=38, height=38, bg=bg, highlightthickness=0)
        logo.grid(row=0, column=0, rowspan=2, padx=(0, 12))
        logo.create_oval(5, 5, 33, 33, outline=gold, width=3)
        for x, y in ((15, 14), (24, 15), (18, 24), (27, 25)):
            logo.create_oval(x - 2, y - 2, x + 2, y + 2, fill=gold, outline=gold)
        ttk.Label(
            header, text="CookieGame", font=("Segoe UI", 24, "bold"),
            style="Header.TLabel",
        ).grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="By gamereal", style="Muted.TLabel").grid(
            row=1, column=1, sticky="w")

        cfg = ttk.Frame(outer, padding=12, style="Section.TFrame")
        cfg.pack(fill="x", pady=(0, 18))
        cfg.columnconfigure(1, weight=1)
        ttk.Label(cfg, text="ADB path:", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Entry(cfg, textvariable=self.adb_path).grid(
            row=0, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(cfg, text="Auto", command=self.auto_adb).grid(
            row=0, column=2, sticky="ew", padx=(8, 0), pady=5)
        ttk.Label(cfg, text="Device:", style="Section.TLabel").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Entry(cfg, textvariable=self.device_serial).grid(
            row=1, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(cfg, text="Check connection", command=self.check_device).grid(
            row=1, column=2, sticky="ew", padx=(8, 0), pady=5)
        ttk.Label(cfg, text="Run count:", style="Section.TLabel").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Entry(cfg, textvariable=self.max_runs, width=12).grid(
            row=2, column=1, sticky="w", padx=8, pady=5)
        ttk.Label(cfg, textvariable=self.device_status, style="SectionMuted.TLabel").grid(
            row=2, column=2, sticky="e", padx=(8, 0), pady=5)
        ttk.Label(cfg, text="0 = unlimited", style="SectionMuted.TLabel").grid(
            row=3, column=1, sticky="w", padx=8, pady=(0, 2))
        ttk.Button(cfg, text="Check boosts", command=self.check_boosts).grid(
            row=3, column=2, sticky="ew", padx=(8, 0), pady=(0, 2))
        ttk.Button(cfg, text="Check action", command=self.check_action).grid(
            row=4, column=2, sticky="ew", padx=(8, 0), pady=(5, 0))

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(0, 10))
        ttk.Label(actions, textvariable=self.status_line, style="Good.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(anchor="center", pady=(0, 10))
        self.action_row = ttk.Frame(actions)
        self.action_row.pack(fill="x")
        self.start_button = tk.Button(
            self.action_row,
            text="Start Bot",
            command=self.start,
            bg="#49b84f",
            fg="#ffffff",
            activebackground="#5fca64",
            activeforeground="#ffffff",
            disabledforeground="#d9d9d9",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=12,
        )
        self.stop_button = tk.Button(
            self.action_row,
            text="Stop Bot",
            command=self.stop,
            state="disabled",
            bg="#ec5353",
            fg="#ffffff",
            activebackground="#ff6b6b",
            activeforeground="#ffffff",
            disabledforeground="#f4d0d0",
            font=("Segoe UI", 12, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=12,
        )
        self._show_start_button()

        summary = ttk.Frame(outer)
        summary.pack(fill="x", pady=(0, 14))
        ttk.Label(summary, textvariable=self.coin_line, style="Coin.TLabel",
                  font=("Segoe UI", 12, "bold")).pack(anchor="center", pady=(6, 0))
        ttk.Label(summary, textvariable=self.rate_line, style="Muted.TLabel").pack(anchor="center")
        ttk.Label(summary, textvariable=self.boost_line, style="Muted.TLabel").pack(anchor="center")
        ttk.Label(summary, textvariable=self.action_line, style="Muted.TLabel").pack(anchor="center")

        ttk.Label(outer, text="Work log:", style="Muted.TLabel").pack(anchor="w")
        log_box = ttk.Frame(outer)
        log_box.pack(fill="both", expand=True, pady=(4, 0))
        self.log = ScrolledText(log_box, height=18, wrap="word", bg="#181824",
                                fg="#f2f2f2", insertbackground="#f2f2f2")
        self.log.pack(fill="both", expand=True)

    def _refresh_summary(self, status: str | None = None) -> None:
        current_status = status if status is not None else self.status.get()
        self.status_line.set(
            format_status_line(current_status, self.run_count.get(), self.device_serial.get()))
        self.coin_line.set(
            format_coin_line(self.latest_coins.get(), self.total_gross.get(), self.total_net.get()))
        self.rate_line.set(format_rate_line(self.net_rate.get(), self.ingredients_rate.get()))

    def _pack_action_button(self, button: tk.Button) -> None:
        button.pack(fill="x", expand=True, padx=210)

    def _show_start_button(self) -> None:
        self.stop_button.pack_forget()
        self.start_button.configure(state="normal")
        self._pack_action_button(self.start_button)

    def _show_stop_button(self, enabled: bool = True) -> None:
        self.start_button.pack_forget()
        self.stop_button.configure(state="normal" if enabled else "disabled")
        self._pack_action_button(self.stop_button)

    def _append_log(self, line: str) -> None:
        self.log.insert("end", line + "\n")
        # Cap the buffer: an overnight unattended session emits tens of thousands of lines,
        # and an unbounded Tk Text widget grows memory and slows every insert. Keep the tail.
        if int(self.log.index("end-1c").split(".")[0]) > 2000:
            self.log.delete("1.0", "end-1000l")
        self.log.see("end")

    def load_defaults(self) -> None:
        try:
            cfg = load_config(self.config_path.get())
        except ConfigError as exc:
            self._append_log(f"[config] {exc}")
            return
        self.capture_backend.set(cfg.capture_backend)
        self.device_serial.set(cfg.device_serial or "")
        self.adb_path.set(discover_adb_path(cfg.adb_path))
        self.allow_boosts.set(cfg.spending.allow_coin_boosts)
        devices = adb_ready_devices(self.adb_path.get())
        selected, status = select_adb_serial(self.device_serial.get(), devices)
        if selected != self.device_serial.get():
            self.device_serial.set(selected)
            self._append_log(f"[adb] switched to {selected}")
        self.device_status.set(status)
        self._refresh_summary("stopped")
        self._append_log(f"[config] loaded {self.config_path.get()}")

    def auto_adb(self) -> None:
        self.adb_path.set(discover_adb_path(self.adb_path.get()))
        self._append_log(f"[adb] {self.adb_path.get() or 'not found'}")

    def check_device(self) -> None:
        adb = self.adb_path.get().strip() or discover_adb_path()
        serial = self.device_serial.get().strip()
        self.device_status.set("checking")

        def worker() -> None:
            try:
                devices = adb_ready_devices(adb)
                selected, status = select_adb_serial(serial, devices)
                if selected != serial:
                    self._events.put(("serial", selected))
                self._events.put(("device", status))
                if status == "device missing":
                    self._events.put(("log", f"[adb] found {devices or 'none'}, not {serial}"))
                elif serial and selected != serial:
                    self._events.put(("log", f"[adb] switched to {selected}"))
                else:
                    self._events.put(("log", f"[adb] devices: {', '.join(devices) or 'none'}"))
            except Exception as exc:
                self._events.put(("device", "check failed"))
                self._events.put(("log", f"[adb] {type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def check_boosts(self) -> None:
        if self._thread and self._thread.is_alive():
            self._append_log("[boost] stop the bot before read-only boost check")
            return
        self.boost_line.set(format_boost_line("checking"))
        cfg_path = self.config_path.get()
        device_serial = self.device_serial.get()
        capture_backend = self.capture_backend.get().strip()
        adb_path = self.adb_path.get()

        def worker() -> None:
            dev = None
            old_adb_path = os.environ.get("ADBUTILS_ADB_PATH")
            try:
                cfg = controller_runtime_config(
                    load_config(cfg_path), device_serial, capture_backend, adb_path)
                if cfg.adb_path:
                    os.environ["ADBUTILS_ADB_PATH"] = cfg.adb_path
                dev = open_device(cfg)
                dev.start()
                frame = dev.last_frame()
                if frame is None:
                    self._events.put(("boost", "no frame"))
                    self._events.put(("log", "[boost] no frame"))
                    return
                status = read_boost_gate_status(frame, TemplateMatcher(cfg.templates_dir))
                detail = format_boost_gate_status(status)
                self._events.put(("boost", "ready" if status.ready_to_play else "not ready"))
                self._events.put(("log", "[boost] " + detail))
            except Exception as exc:
                self._events.put(("boost", "check failed"))
                self._events.put(("log", f"[boost] {type(exc).__name__}: {exc}"))
            finally:
                try:
                    if dev is not None:
                        dev.stop()
                finally:
                    if adb_path:
                        if old_adb_path is None:
                            os.environ.pop("ADBUTILS_ADB_PATH", None)
                        else:
                            os.environ["ADBUTILS_ADB_PATH"] = old_adb_path

        threading.Thread(target=worker, daemon=True).start()

    def check_action(self) -> None:
        if self._thread and self._thread.is_alive():
            self._append_log("[action-check] stop the bot before read-only action check")
            return
        self.action_line.set(format_action_line("checking"))
        cfg_path = self.config_path.get()
        device_serial = self.device_serial.get()
        capture_backend = self.capture_backend.get().strip()
        adb_path = self.adb_path.get()

        def worker() -> None:
            dev = None
            old_adb_path = os.environ.get("ADBUTILS_ADB_PATH")
            try:
                cfg = controller_runtime_config(
                    load_config(cfg_path), device_serial, capture_backend, adb_path)
                if cfg.adb_path:
                    os.environ["ADBUTILS_ADB_PATH"] = cfg.adb_path
                dev = open_device(cfg)
                dev.start()
                frame = dev.last_frame()
                if frame is None:
                    self._events.put(("action", "no frame"))
                    self._events.put(("log", "[action-check] no frame"))
                    return
                agent = StreamingRuleBasedAgent(cfg)
                agent.reset()
                matcher = TemplateMatcher(cfg.templates_dir)
                sample = read_action_sample(
                    frame, cfg, agent, 1, None, 0.0, matcher=matcher)
                self._events.put(("action", "advisor " + format_action_brief(sample)))
                self._events.put(("log", "[action-check] " + format_action_sample(sample)))
            except Exception as exc:
                self._events.put(("action", "check failed"))
                self._events.put(("log", f"[action-check] {type(exc).__name__}: {exc}"))
            finally:
                try:
                    if dev is not None:
                        dev.stop()
                finally:
                    if adb_path:
                        if old_adb_path is None:
                            os.environ.pop("ADBUTILS_ADB_PATH", None)
                        else:
                            os.environ["ADBUTILS_ADB_PATH"] = old_adb_path

        threading.Thread(target=worker, daemon=True).start()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            max_runs = parse_max_runs(self.max_runs.get())
        except ValueError as exc:
            self._append_log(f"[ui] {exc}")
            return
        capture_backend = self.capture_backend.get().strip()
        if capture_backend not in CAPTURE_BACKENDS:
            self._append_log(f"[ui] unknown capture backend: {capture_backend}")
            return
        self._stop_event = threading.Event()
        self.status.set("running")
        self._show_stop_button()
        self.run_count.set("0")
        self.latest_coins.set("-")
        self.total_gross.set("0")
        self.total_net.set("0")
        self.net_rate.set("0/hr")
        self.ingredients_rate.set("0/hr")
        self.action_line.set(format_action_line(""))
        self._refresh_summary("running")
        cfg_path = self.config_path.get()
        allow_boosts = self.allow_boosts.get()
        adb_path = self.adb_path.get()
        device_serial = self.device_serial.get()

        def worker() -> None:
            try:
                farm(
                    cfg_path,
                    max_runs=max_runs,
                    stop_event=self._stop_event,
                    log=lambda msg: self._events.put(("log", msg)),
                    on_result=lambda run, result, metrics: self._events.put(
                        ("result", run, result, metrics)
                    ),
                    allow_coin_boosts=allow_boosts,
                    device_serial=device_serial,
                    capture_backend=capture_backend,
                    adb_path=adb_path,
                )
            except Exception as exc:
                self._events.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                self._events.put(("done",))

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self.status.set("stopping")
        self._refresh_summary("stopping")
        self._show_stop_button(enabled=False)

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._stop_event is not None:
            self._stop_event.set()
        self._join_then_close()

    def _join_then_close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=0.1)
            if self._thread.is_alive():
                self.root.after(100, self._join_then_close)
                return
        self.root.destroy()

    def _poll_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "log":
                line = str(event[1])
                self._append_log(line)
                action = summarize_action_log(line)
                if action:
                    self.action_line.set(format_action_line(action))
                boost = summarize_boost_log(line)
                if boost:
                    self.boost_line.set(format_boost_line(boost))
            elif kind == "result":
                _, run, result, metrics = event
                self.run_count.set(str(run))
                self.latest_coins.set(str(result.coins))
                self.total_gross.set(str(metrics.total_coins()))
                self.total_net.set(str(metrics.total_net_coins()))
                self.net_rate.set(f"{metrics.net_coins_per_hour():.0f}/hr")
                self.ingredients_rate.set(f"{metrics.ingredients_per_hour():.1f}/hr")
                self._refresh_summary("running")
            elif kind == "error":
                self._append_log("[error] " + str(event[1]))
                self._refresh_summary("error")
            elif kind == "device":
                self.device_status.set(str(event[1]))
                self._refresh_summary()
            elif kind == "serial":
                self.device_serial.set(str(event[1]))
                self._refresh_summary()
            elif kind == "boost":
                self.boost_line.set(format_boost_line(str(event[1])))
            elif kind == "action":
                self.action_line.set(format_action_line(str(event[1])))
            elif kind == "done":
                self.status.set("stopped")
                self._refresh_summary("stopped")
                self._show_start_button()
        self.root.after(100, self._poll_events)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    ControllerApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
