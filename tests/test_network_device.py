import numpy as np
import cv2
import pytest
from cookierun_bot.device import BridgeCommandError, NetworkDevice


class FakeSock:
    def __init__(self, to_read=b""):
        self._read = bytearray(to_read)
        self.sent = bytearray()
        self.closed = False

    def sendall(self, b):
        self.sent += b

    def recv(self, n):
        chunk = bytes(self._read[:n])
        del self._read[:n]
        return chunk

    def setsockopt(self, *a):
        pass

    def close(self):
        self.closed = True


def _nd(fake):
    nd = NetworkDevice("host")
    nd._sock = fake
    return nd


def test_tap_sends_command_and_reads_ok():
    fake = FakeSock(b"OK\n")
    _nd(fake).tap(12, 34)
    assert fake.sent == b"TAP 12 34\n"


def test_hold_sends_duration_and_reads_ok():
    fake = FakeSock(b"OK\n")
    _nd(fake).hold(5, 6, 300)
    assert fake.sent == b"HOLD 5 6 300\n"


def test_landscape_tap_uses_captured_screen_coordinates_without_rotation():
    fake = FakeSock(b"OK\n")
    nd = _nd(fake)
    nd._cap = (2560, 1440)
    nd._rot = 1

    nd.tap(2000, 1200)

    assert fake.sent == b"TAP 2000 1200\n"


def test_start_authenticates_before_sending_commands(monkeypatch):
    fake = FakeSock(b"OK\n")
    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: fake)

    nd = NetworkDevice("phone", token="copy-from-phone")
    nd.start()

    assert fake.sent == b"AUTH copy-from-phone\n"


def test_start_requires_bridge_token(monkeypatch):
    monkeypatch.delenv("COOKIERUN_BRIDGE_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="COOKIERUN_BRIDGE_TOKEN"):
        NetworkDevice("phone").start()


def test_start_closes_socket_when_authentication_is_rejected(monkeypatch):
    fake = FakeSock(b"ERR auth\n")
    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: fake)

    with pytest.raises(BridgeCommandError, match="authentication failed"):
        NetworkDevice("phone", token="wrong-token").start()

    assert fake.closed


def test_tap_raises_when_bridge_gesture_failed():
    fake = FakeSock(b"OK no_acc\n")
    with pytest.raises(BridgeCommandError):
        _nd(fake).tap(12, 34)


def test_last_frame_none_when_zero_length():
    fake = FakeSock((0).to_bytes(4, "big"))
    nd = _nd(fake)
    assert nd.last_frame() is None
    assert fake.sent == b"FRAME\n"


def test_last_frame_decodes_jpeg_and_sets_resolution():
    img = np.zeros((20, 30, 3), np.uint8)
    img[:] = (0, 0, 255)
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    fake = FakeSock(len(jpeg).to_bytes(4, "big") + jpeg)
    nd = _nd(fake)
    out = nd.last_frame()
    assert out is not None and out.shape == (20, 30, 3)
    assert nd.resolution == (30, 20)
