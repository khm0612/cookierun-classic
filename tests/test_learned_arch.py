import pytest

torch = pytest.importorskip("torch")

from cookierun_bot.policies.learned import build_net_from_meta


def _meta(arch):
    return {
        "classes": ["none", "jump", "slide"],
        "arch": arch,
        "K": 4,
        "H": 96,
        "W": 224,
        "conv": [[8, 3, 2]],
        "fc": 16,
    }


def _first_conv(module):
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            return child
    raise AssertionError("no Conv2d found")


def test_small_cnn_still_builds_from_legacy_meta():
    net = build_net_from_meta(torch, {k: v for k, v in _meta("small_cnn").items() if k != "arch"})
    out = net(torch.zeros(2, 4, 96, 224))
    assert out.shape == (2, 3)


@pytest.mark.parametrize("arch", ["mobilenet_v3_large", "efficientnet_b5"])
def test_torchvision_arches_accept_k_frame_input(arch):
    pytest.importorskip("torchvision")
    net = build_net_from_meta(torch, _meta(arch))
    assert _first_conv(net).in_channels == 4
    assert net.classifier[-1].out_features == 3
