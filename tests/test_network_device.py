import numpy as np
import cv2
import pytest
from cookierun_bot.device import BridgeCommandError, NetworkDevice


class FakeSock:
    def __init__(self, to_read=b""):
        self._read = bytearray(to_read)
        self.sent = bytearray()

    def sendall(self, b):
        self.sent += b

    def recv(self, n):
        chunk = bytes(self._read[:n])
        del self._read[:n]
        return chunk

    def setsockopt(self, *a):
        pass

    def close(self):
        pass


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
