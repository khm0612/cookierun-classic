from __future__ import annotations

ACTION_NOOP = 0
ACTION_JUMP = 1
ACTION_SLIDE = 2
N_ACTIONS = 3


def apply_action(device, action: int, g) -> None:
    if action == ACTION_JUMP:
        # holding Jump jumps higher/longer than a tap — use a held press when configured
        hold_ms = getattr(g, "jump_hold_ms", 0)
        if hold_ms > 0:
            device.hold(g.jump_button[0], g.jump_button[1], hold_ms)
        else:
            device.tap(*g.jump_button)
    elif action == ACTION_SLIDE:
        device.hold(g.slide_button[0], g.slide_button[1], g.slide_hold_ms)
    # ACTION_NOOP: intentionally do nothing
