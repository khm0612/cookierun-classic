from cookierun_bot.win_input import map_guest_to_screen


def test_map_confirm_button_matches_validated_click():
    # Real BlueStacks case that we clicked successfully: window rect + chrome,
    # guest 1920x1080, Confirm button centre ~ (960, 690) -> screen ~ (5392, 666).
    sx, sy = map_guest_to_screen((4702, 131, 6123, 945), 40, 40, 1920, 1080, 960, 690)
    assert abs(sx - 5392) <= 2
    assert abs(sy - 666) <= 2


def test_map_top_left_origin_no_letterbox():
    # game area exactly 1920x1080 (window 1960x1120 minus 40 chrome each) -> scale 1.
    sx, sy = map_guest_to_screen((0, 0, 1960, 1120), 40, 40, 1920, 1080, 0, 0)
    assert (sx, sy) == (0, 40)


def test_map_center_scales_linearly():
    sx, sy = map_guest_to_screen((0, 0, 1960, 1120), 40, 40, 1920, 1080, 1920, 1080)
    assert (sx, sy) == (1920, 1120)
