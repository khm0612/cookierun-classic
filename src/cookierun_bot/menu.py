from __future__ import annotations


_ALLOWLIST_ALIASES = {
    "play": ("start", "restart", "replay"),
    "openall": ("collect",),
}


class MenuNavigator:
    def __init__(self, device, matcher, cfg):
        self._device = device
        self._matcher = matcher
        self._cfg = cfg

    def is_spend_dialog(self, frame) -> bool:
        denied = list(self._cfg.menu_denylist)
        if self._cfg.spending.forbid_crystals and "revive_crystals" not in denied:
            denied.append("revive_crystals")
        return any(self._matcher.present(frame, name) for name in denied)

    def is_allowed(self, name: str) -> bool:
        allowed = self._cfg.menu_allowlist
        return name in allowed or any(
            alias in allowed for alias in _ALLOWLIST_ALIASES.get(name, ())
        )

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
