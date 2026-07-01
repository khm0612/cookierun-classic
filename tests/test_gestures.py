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
